"""Factory for creating trainers based on configuration."""

import logging
from typing import Union

from datasets import Dataset
from omegaconf import DictConfig

from ..models import LocalModel
from .dpo_trainer import DPOTrainerWrapper
from .sft_trainer import SFTTrainerWrapper

logger = logging.getLogger(__name__)

TrainerType = Union[SFTTrainerWrapper, DPOTrainerWrapper]


def create_trainer(
    cfg: DictConfig,
    model: LocalModel,
    train_dataset: Dataset,
    eval_dataset: Dataset | None = None,
    ref_model: LocalModel | None = None,
) -> TrainerType:
    """Create appropriate trainer based on configuration.

    Args:
        cfg: Full configuration.
        model: LocalModel instance.
        train_dataset: Training dataset.
        eval_dataset: Optional evaluation dataset.
        ref_model: Optional reference model for DPO.

    Returns:
        Trainer instance.

    Raises:
        ValueError: If training type is not supported.
    """
    training_type = cfg.training.type

    if training_type == "sft":
        logger.info(f"Creating SFT trainer with method: {cfg.training.method}")
        return SFTTrainerWrapper(
            cfg=cfg,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    elif training_type == "dpo":
        logger.info("Creating DPO trainer")
        return DPOTrainerWrapper(
            cfg=cfg,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            ref_model=ref_model,
        )

    else:
        raise ValueError(
            f"Unknown training type: {training_type}. "
            f"Supported types: sft, dpo"
        )


def get_trainer_for_method(method: str) -> type:
    """Get trainer class for a given method.

    Args:
        method: Training method name.

    Returns:
        Trainer class.
    """
    method_to_trainer = {
        "sft": SFTTrainerWrapper,
        "lora": SFTTrainerWrapper,
        "qlora": SFTTrainerWrapper,
        "prefix_tuning": SFTTrainerWrapper,
        "dpo": DPOTrainerWrapper,
    }

    if method not in method_to_trainer:
        raise ValueError(
            f"Unknown training method: {method}. "
            f"Supported: {list(method_to_trainer.keys())}"
        )

    return method_to_trainer[method]
