"""DPO (Direct Preference Optimization) trainer."""

import logging
from pathlib import Path
from typing import Any

import mlflow
import torch
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer

from ..models import LocalModel, get_device

logger = logging.getLogger(__name__)


class DPOTrainerWrapper:
    """DPO trainer with PEFT and MLflow integration."""

    def __init__(
        self,
        cfg: DictConfig,
        model: LocalModel,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        ref_model: LocalModel | None = None,
    ):
        """Initialize DPO trainer.

        Args:
            cfg: Full configuration.
            model: LocalModel instance for policy model.
            train_dataset: Training dataset with prompt/chosen/rejected.
            eval_dataset: Optional evaluation dataset.
            ref_model: Optional reference model (for non-reference-free DPO).
        """
        self.cfg = cfg
        self.model_wrapper = model
        self.ref_model_wrapper = ref_model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self._model: PreTrainedModel | None = None
        self._ref_model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizer | None = None
        self._trainer: DPOTrainer | None = None

    def setup(self) -> None:
        """Set up models and configuration."""
        self.model_wrapper.load()
        self._model = self.model_wrapper.get_model_for_training()
        self._tokenizer = self.model_wrapper.get_tokenizer()

        training_cfg = self.cfg.training
        dpo_cfg = training_cfg.dpo

        lora_cfg = training_cfg.get("lora", {})
        if lora_cfg.get("enabled", True):
            self._model = self._setup_lora()

        if not dpo_cfg.reference_free:
            if self.ref_model_wrapper:
                self.ref_model_wrapper.load()
                self._ref_model = self.ref_model_wrapper.get_model_for_training()
            else:
                logger.info("Using policy model as reference (will be frozen)")
                self._ref_model = None

    def _setup_lora(self) -> PreTrainedModel:
        """Configure LoRA for DPO training."""
        lora_cfg = self.cfg.training.lora

        quant_cfg = self.cfg.training.get("quantization", {})
        if quant_cfg.get("enabled", False):
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
        logger.info(f"LoRA configured for DPO with r={lora_cfg.r}")
        return model

    def _get_training_args(self) -> DPOConfig:
        """Create DPOConfig from configuration."""
        training_cfg = self.cfg.training.training
        logging_cfg = self.cfg.training.logging
        dpo_cfg = self.cfg.training.dpo

        device = get_device()
        output_dir = Path(self.cfg.project.output_dir) / "checkpoints"

        config = DPOConfig(
            output_dir=str(output_dir),
            num_train_epochs=training_cfg.num_epochs,
            per_device_train_batch_size=training_cfg.batch_size,
            per_device_eval_batch_size=training_cfg.batch_size,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            learning_rate=training_cfg.learning_rate,
            weight_decay=training_cfg.weight_decay,
            warmup_ratio=training_cfg.warmup_ratio,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
            logging_steps=logging_cfg.log_steps,
            eval_steps=logging_cfg.eval_steps if self.eval_dataset else None,
            save_steps=logging_cfg.save_steps,
            save_total_limit=logging_cfg.save_total_limit,
            eval_strategy="steps" if self.eval_dataset else "no",
            load_best_model_at_end=True if self.eval_dataset else False,
            report_to="mlflow",
            fp16=device == "cuda",
            remove_unused_columns=False,
            seed=self.cfg.seed,
            beta=dpo_cfg.beta,
            loss_type=dpo_cfg.loss_type,
            label_smoothing=dpo_cfg.label_smoothing,
            max_length=training_cfg.max_seq_length,
            max_prompt_length=training_cfg.get("max_prompt_length", 512),
        )

        return config

    def train(self) -> dict[str, Any]:
        """Run DPO training.

        Returns:
            Training metrics.
        """
        if self._model is None:
            self.setup()

        training_args = self._get_training_args()

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

        with mlflow.start_run(run_name=f"{self.cfg.mlflow.run_name}_dpo"):
            mlflow.log_params(OmegaConf.to_container(self.cfg.training, resolve=True))
            mlflow.log_params({
                "model": self.cfg.model.name,
                "training_type": "dpo",
            })

            self._trainer = DPOTrainer(
                model=self._model,
                ref_model=self._ref_model,
                args=training_args,
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

            logger.info(f"DPO training complete. Model saved to {final_path}")

        return metrics

    def evaluate(self) -> dict[str, float]:
        """Evaluate on the eval dataset.

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
