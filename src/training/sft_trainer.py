"""Supervised Fine-Tuning trainer with PEFT support."""

import logging
from pathlib import Path
from typing import Any

import mlflow
import torch
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

from ..models import LocalModel, get_device

logger = logging.getLogger(__name__)


class SFTTrainerWrapper:
    """Supervised Fine-Tuning trainer with PEFT and MLflow integration."""

    def __init__(
        self,
        cfg: DictConfig,
        model: LocalModel,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
    ):
        """Initialize SFT trainer.

        Args:
            cfg: Full configuration.
            model: LocalModel instance.
            train_dataset: Training dataset.
            eval_dataset: Optional evaluation dataset.
        """
        self.cfg = cfg
        self.model_wrapper = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizer | None = None
        self._trainer: Trainer | None = None

    def setup(self) -> None:
        """Set up model, tokenizer, and PEFT configuration."""
        self.model_wrapper.load()
        self._model = self.model_wrapper.get_model_for_training()
        self._tokenizer = self.model_wrapper.get_tokenizer()

        training_cfg = self.cfg.training

        if training_cfg.method in ["lora", "qlora"]:
            self._model = self._setup_lora()
        elif training_cfg.method == "prefix_tuning":
            self._model = self._setup_prefix_tuning()

        self._model.print_trainable_parameters()

    def _setup_lora(self) -> PreTrainedModel:
        """Configure LoRA/QLoRA for the model."""
        training_cfg = self.cfg.training
        lora_cfg = training_cfg.lora

        if training_cfg.method == "qlora":
            self._model = prepare_model_for_kbit_training(self._model)

        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            target_modules=list(lora_cfg.target_modules),
            bias=lora_cfg.bias,
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(self._model, peft_config)
        logger.info(f"LoRA configured with r={lora_cfg.r}, alpha={lora_cfg.lora_alpha}")
        return model

    def _setup_prefix_tuning(self) -> PreTrainedModel:
        """Configure Prefix Tuning for the model."""
        prefix_cfg = self.cfg.training.prefix

        peft_config = PrefixTuningConfig(
            num_virtual_tokens=prefix_cfg.num_virtual_tokens,
            encoder_hidden_size=prefix_cfg.encoder_hidden_size,
            prefix_projection=prefix_cfg.prefix_projection,
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(self._model, peft_config)
        logger.info(
            f"Prefix Tuning configured with {prefix_cfg.num_virtual_tokens} tokens"
        )
        return model

    def _get_training_args(self) -> TrainingArguments:
        """Create TrainingArguments from configuration."""
        training_cfg = self.cfg.training.training
        logging_cfg = self.cfg.training.logging
        optimizer_cfg = self.cfg.training.optimizer

        device = get_device()
        output_dir = Path(self.cfg.project.output_dir) / "checkpoints"

        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=training_cfg.num_epochs,
            per_device_train_batch_size=training_cfg.batch_size,
            per_device_eval_batch_size=training_cfg.batch_size,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            learning_rate=training_cfg.learning_rate,
            weight_decay=training_cfg.weight_decay,
            warmup_ratio=training_cfg.warmup_ratio,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
            optim=optimizer_cfg.name,
            logging_steps=logging_cfg.log_steps,
            eval_steps=logging_cfg.eval_steps if self.eval_dataset else None,
            save_steps=logging_cfg.save_steps,
            save_total_limit=logging_cfg.save_total_limit,
            eval_strategy="steps" if self.eval_dataset else "no",
            load_best_model_at_end=True if self.eval_dataset else False,
            report_to="mlflow",
            fp16=device == "cuda",
            bf16=False,
            dataloader_num_workers=self.cfg.hardware.num_workers,
            remove_unused_columns=False,
            seed=self.cfg.seed,
        )

        return args

    def train(self) -> dict[str, Any]:
        """Run the training loop.

        Returns:
            Training metrics.
        """
        if self._model is None:
            self.setup()

        training_args = self._get_training_args()
        max_seq_length = self.cfg.training.training.max_seq_length

        callbacks = []
        if self.cfg.training.early_stopping.enabled and self.eval_dataset:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.cfg.training.early_stopping.patience,
                    early_stopping_threshold=self.cfg.training.early_stopping.threshold,
                )
            )

        mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow.experiment_name)

        with mlflow.start_run(run_name=self.cfg.mlflow.run_name):
            mlflow.log_params(OmegaConf.to_container(self.cfg.training, resolve=True))
            mlflow.log_params({"model": self.cfg.model.name})

            sft_config = SFTConfig(
                **{k: v for k, v in training_args.to_dict().items()
                   if k not in ['_n_gpu', 'parallel_mode']},
                max_length=max_seq_length,
                packing=False,
                dataset_text_field="text",
            )

            self._trainer = SFTTrainer(
                model=self._model,
                args=sft_config,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                processing_class=self._tokenizer,
                callbacks=callbacks,
            )

            train_result = self._trainer.train()

            metrics = train_result.metrics
            mlflow.log_metrics(metrics)

            final_path = Path(training_args.output_dir) / "final"
            self._trainer.save_model(str(final_path))
            self._tokenizer.save_pretrained(str(final_path))
            mlflow.log_artifacts(str(final_path), artifact_path="model")

            logger.info(f"Training complete. Model saved to {final_path}")

        return metrics

    def evaluate(self) -> dict[str, float]:
        """Evaluate the model on the eval dataset.

        Returns:
            Evaluation metrics.
        """
        if self._trainer is None:
            raise RuntimeError("Call train() before evaluate()")

        if self.eval_dataset is None:
            raise ValueError("No evaluation dataset provided")

        return self._trainer.evaluate()

    def save_model(self, path: str | Path) -> None:
        """Save the trained model.

        Args:
            path: Output path.
        """
        if self._trainer is None:
            raise RuntimeError("Call train() before save_model()")

        path = Path(path)
        self._trainer.save_model(str(path))
        self._tokenizer.save_pretrained(str(path))
        logger.info(f"Model saved to {path}")
