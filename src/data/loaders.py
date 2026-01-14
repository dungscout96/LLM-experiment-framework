"""DataLoader utilities for LLM training."""

import logging
from typing import Any

from datasets import Dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


def create_sft_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    cfg: DictConfig,
    is_train: bool = True,
) -> DataLoader:
    """Create DataLoader for SFT training.

    Args:
        dataset: HuggingFace Dataset.
        tokenizer: Tokenizer for collation.
        cfg: Data configuration.
        is_train: Whether this is training data.

    Returns:
        PyTorch DataLoader.
    """
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    dataloader_cfg = cfg.dataloader
    batch_size = cfg.get("training", {}).get("batch_size", 4)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataloader_cfg.shuffle_train if is_train else False,
        collate_fn=collator,
        num_workers=cfg.get("hardware", {}).get("num_workers", 4),
        pin_memory=dataloader_cfg.pin_memory,
        drop_last=dataloader_cfg.drop_last if is_train else False,
    )


def create_seq2seq_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    cfg: DictConfig,
    is_train: bool = True,
) -> DataLoader:
    """Create DataLoader for sequence-to-sequence training.

    Args:
        dataset: HuggingFace Dataset.
        tokenizer: Tokenizer for collation.
        cfg: Data configuration.
        is_train: Whether this is training data.

    Returns:
        PyTorch DataLoader.
    """
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    dataloader_cfg = cfg.dataloader
    batch_size = cfg.get("training", {}).get("batch_size", 4)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataloader_cfg.shuffle_train if is_train else False,
        collate_fn=collator,
        num_workers=cfg.get("hardware", {}).get("num_workers", 4),
        pin_memory=dataloader_cfg.pin_memory,
        drop_last=dataloader_cfg.drop_last if is_train else False,
    )


class DPODataCollator:
    """Data collator for DPO training."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_prompt_length: int = 256,
    ):
        """Initialize DPO data collator.

        Args:
            tokenizer: Tokenizer for encoding.
            max_length: Maximum total sequence length.
            max_prompt_length: Maximum prompt length.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch of DPO examples.

        Args:
            features: List of examples with prompt, chosen, rejected.

        Returns:
            Batched and tokenized examples.
        """
        batch = {
            "prompt": [f["prompt"] for f in features],
            "chosen": [f["chosen"] for f in features],
            "rejected": [f["rejected"] for f in features],
        }

        prompt_tokens = self.tokenizer(
            batch["prompt"],
            truncation=True,
            max_length=self.max_prompt_length,
            padding=True,
            return_tensors="pt",
        )

        chosen_tokens = self.tokenizer(
            [p + c for p, c in zip(batch["prompt"], batch["chosen"])],
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )

        rejected_tokens = self.tokenizer(
            [p + r for p, r in zip(batch["prompt"], batch["rejected"])],
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )

        return {
            "prompt_input_ids": prompt_tokens["input_ids"],
            "prompt_attention_mask": prompt_tokens["attention_mask"],
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"],
        }


def create_dpo_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    cfg: DictConfig,
    is_train: bool = True,
) -> DataLoader:
    """Create DataLoader for DPO training.

    Args:
        dataset: HuggingFace Dataset with prompt/chosen/rejected.
        tokenizer: Tokenizer for collation.
        cfg: Training configuration.
        is_train: Whether this is training data.

    Returns:
        PyTorch DataLoader.
    """
    training_cfg = cfg.training
    collator = DPODataCollator(
        tokenizer=tokenizer,
        max_length=training_cfg.max_seq_length,
        max_prompt_length=training_cfg.get("max_prompt_length", 256),
    )

    dataloader_cfg = cfg.get("data", {}).get("dataloader", {})
    batch_size = training_cfg.batch_size

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataloader_cfg.get("shuffle_train", True) if is_train else False,
        collate_fn=collator,
        num_workers=cfg.get("hardware", {}).get("num_workers", 4),
        pin_memory=dataloader_cfg.get("pin_memory", True),
        drop_last=True if is_train else False,
    )


def get_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    cfg: DictConfig,
    training_type: str = "sft",
    is_train: bool = True,
) -> DataLoader:
    """Factory function to create the appropriate DataLoader.

    Args:
        dataset: HuggingFace Dataset.
        tokenizer: Tokenizer for collation.
        cfg: Configuration.
        training_type: Type of training (sft, dpo).
        is_train: Whether this is training data.

    Returns:
        PyTorch DataLoader.
    """
    if training_type == "dpo":
        return create_dpo_dataloader(dataset, tokenizer, cfg, is_train)
    else:
        return create_sft_dataloader(dataset, tokenizer, cfg, is_train)
