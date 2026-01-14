"""Dataset classes for LLM training."""

import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from .preprocessor import DataPreprocessor, apply_template

logger = logging.getLogger(__name__)


class LLMDataset:
    """Dataset wrapper for LLM fine-tuning."""

    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: PreTrainedTokenizer | None = None,
    ):
        """Initialize dataset with configuration.

        Args:
            cfg: Data configuration from Hydra.
            tokenizer: Optional tokenizer for preprocessing.
        """
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.preprocessor = DataPreprocessor(cfg)
        self._dataset: DatasetDict | None = None

    def load(self) -> DatasetDict:
        """Load and prepare the dataset.

        Returns:
            HuggingFace DatasetDict with train/val/test splits.
        """
        paths = self.cfg.paths

        if paths.train_file:
            train_df = self.preprocessor.load_file(paths.train_file)
            self.preprocessor.validate_columns(train_df)
            train_df = self.preprocessor.preprocess(train_df)

            if paths.val_file:
                val_df = self.preprocessor.load_file(paths.val_file)
                val_df = self.preprocessor.preprocess(val_df)
            else:
                val_df = None

            if paths.test_file:
                test_df = self.preprocessor.load_file(paths.test_file)
                test_df = self.preprocessor.preprocess(test_df)
            else:
                test_df = None

            self._dataset = self._create_splits(train_df, val_df, test_df)
        else:
            raise ValueError("train_file must be specified in data configuration")

        logger.info(f"Dataset loaded: {self._dataset}")
        return self._dataset

    def _create_splits(
        self,
        train_df,
        val_df=None,
        test_df=None,
    ) -> DatasetDict:
        """Create train/val/test splits from DataFrames.

        Args:
            train_df: Training DataFrame.
            val_df: Optional validation DataFrame.
            test_df: Optional test DataFrame.

        Returns:
            DatasetDict with splits.
        """
        split_cfg = self.cfg.split
        datasets = {}

        if val_df is None and test_df is None:
            n_samples = len(train_df)
            min_samples_for_split = 10

            if n_samples < min_samples_for_split:
                logger.warning(
                    f"Dataset too small ({n_samples} samples) to split. "
                    f"Using all data for training."
                )
                datasets["train"] = Dataset.from_pandas(train_df, preserve_index=False)
                return DatasetDict(datasets)

            from sklearn.model_selection import train_test_split

            train_ratio = split_cfg.train_ratio
            val_ratio = split_cfg.val_ratio
            test_ratio = split_cfg.test_ratio

            val_test_ratio = val_ratio + test_ratio
            if val_test_ratio > 0 and n_samples * val_test_ratio < 2:
                logger.warning(
                    f"Not enough samples for val/test split. Using 90/10 train/val."
                )
                train_data, val_data = train_test_split(
                    train_df,
                    test_size=max(1, int(n_samples * 0.1)),
                    shuffle=split_cfg.shuffle,
                    random_state=42,
                )
                datasets["train"] = Dataset.from_pandas(train_data, preserve_index=False)
                datasets["validation"] = Dataset.from_pandas(val_data, preserve_index=False)
                return DatasetDict(datasets)

            train_data, temp_data = train_test_split(
                train_df,
                train_size=train_ratio,
                shuffle=split_cfg.shuffle,
                random_state=42,
            )

            if test_ratio > 0 and len(temp_data) >= 2:
                relative_val = val_ratio / (val_ratio + test_ratio)
                val_data, test_data = train_test_split(
                    temp_data,
                    train_size=relative_val,
                    shuffle=split_cfg.shuffle,
                    random_state=42,
                )
                datasets["test"] = Dataset.from_pandas(test_data, preserve_index=False)
            else:
                val_data = temp_data

            datasets["train"] = Dataset.from_pandas(train_data, preserve_index=False)
            datasets["validation"] = Dataset.from_pandas(val_data, preserve_index=False)
        else:
            datasets["train"] = Dataset.from_pandas(train_df, preserve_index=False)
            if val_df is not None:
                datasets["validation"] = Dataset.from_pandas(val_df, preserve_index=False)
            if test_df is not None:
                datasets["test"] = Dataset.from_pandas(test_df, preserve_index=False)

        return DatasetDict(datasets)

    def prepare_for_training(
        self,
        training_type: str = "sft",
        max_seq_length: int = 2048,
    ) -> DatasetDict:
        """Prepare dataset for training with tokenization.

        Args:
            training_type: Type of training (sft, dpo).
            max_seq_length: Maximum sequence length.

        Returns:
            Tokenized DatasetDict ready for training.
        """
        if self._dataset is None:
            self.load()

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be set before preparing for training")

        if training_type == "dpo":
            return self._prepare_dpo_dataset(max_seq_length)
        else:
            return self._prepare_sft_dataset(max_seq_length)

    def _prepare_sft_dataset(self, max_seq_length: int) -> DatasetDict:
        """Prepare dataset for SFT training."""

        def format_example(example):
            template = getattr(self.cfg, "prompt_template", None)
            template_no_input = getattr(self.cfg, "prompt_template_no_input", None)

            if template:
                text = apply_template(
                    {
                        "instruction": example.get(self.cfg.columns.instruction, ""),
                        "input": example.get(self.cfg.columns.input, ""),
                        "output": example.get(self.cfg.columns.output, ""),
                    },
                    template,
                    template_no_input,
                )
            else:
                instruction = example.get(self.cfg.columns.instruction, "")
                input_text = example.get(self.cfg.columns.input, "")
                output = example.get(self.cfg.columns.output, "")

                if input_text:
                    text = f"{instruction}\n\n{input_text}\n\n{output}"
                else:
                    text = f"{instruction}\n\n{output}"

            return {"text": text}

        formatted = self._dataset.map(format_example)

        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_length,
                padding=False,
            )

        tokenized = formatted.map(
            tokenize,
            batched=True,
            remove_columns=formatted["train"].column_names,
        )

        return tokenized

    def _prepare_dpo_dataset(self, max_seq_length: int) -> DatasetDict:
        """Prepare dataset for DPO training."""

        def format_dpo(example):
            prompt = example.get(self.cfg.columns.instruction, "")
            chosen = example.get(self.cfg.columns.chosen, "")
            rejected = example.get(self.cfg.columns.rejected, "")

            return {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }

        return self._dataset.map(format_dpo)

    @property
    def dataset(self) -> DatasetDict | None:
        """Get the underlying HuggingFace dataset."""
        return self._dataset


class ChatDataset(LLMDataset):
    """Dataset for chat-format fine-tuning."""

    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: PreTrainedTokenizer | None = None,
        system_message: str | None = None,
    ):
        """Initialize chat dataset.

        Args:
            cfg: Data configuration.
            tokenizer: Tokenizer for preprocessing.
            system_message: Optional system message.
        """
        super().__init__(cfg, tokenizer)
        self.system_message = system_message

    def _prepare_sft_dataset(self, max_seq_length: int) -> DatasetDict:
        """Prepare chat dataset for SFT training."""

        def format_chat(example):
            messages = []

            if self.system_message:
                messages.append({"role": "system", "content": self.system_message})

            instruction = example.get(self.cfg.columns.instruction, "")
            input_text = example.get(self.cfg.columns.input, "")
            output = example.get(self.cfg.columns.output, "")

            user_content = instruction
            if input_text:
                user_content += f"\n\n{input_text}"

            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": output})

            if hasattr(self.tokenizer, "apply_chat_template"):
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                text = "\n".join(
                    f"{m['role']}: {m['content']}" for m in messages
                )

            return {"text": text}

        formatted = self._dataset.map(format_chat)

        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_length,
                padding=False,
            )

        tokenized = formatted.map(
            tokenize,
            batched=True,
            remove_columns=formatted["train"].column_names,
        )

        return tokenized
