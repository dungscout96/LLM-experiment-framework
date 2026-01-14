#!/usr/bin/env python3
"""Main training entry point with Hydra configuration."""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.logging import RichHandler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import LLMDataset
from src.models import LocalModel, ModelFactory
from src.training import create_trainer

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Set up logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run training with Hydra configuration.

    Args:
        cfg: Hydra configuration.
    """
    setup_logging(cfg.get("verbose", False))
    logger = logging.getLogger(__name__)

    console.print("\n[bold blue]LLM Experiment Framework[/bold blue]")
    console.print(f"Model: [green]{cfg.model.name}[/green]")
    console.print(f"Training: [green]{cfg.training.method}[/green]")
    console.print(f"Device: [green]{cfg.hardware.device}[/green]\n")

    if cfg.get("print_config", False):
        console.print(OmegaConf.to_yaml(cfg))

    logger.info("Loading dataset...")
    dataset = LLMDataset(cfg.data)
    dataset.load()

    logger.info("Loading model...")
    model = LocalModel(cfg.model)

    tokenizer = model.get_tokenizer()
    dataset.tokenizer = tokenizer
    prepared_dataset = dataset.prepare_for_training(
        training_type=cfg.training.type,
        max_seq_length=cfg.training.training.max_seq_length,
    )

    train_data = prepared_dataset["train"]
    eval_data = prepared_dataset.get("validation")

    logger.info(f"Train samples: {len(train_data)}")
    if eval_data:
        logger.info(f"Eval samples: {len(eval_data)}")

    logger.info("Creating trainer...")
    trainer = create_trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
    )

    logger.info("Starting training...")
    with console.status("[bold green]Training in progress..."):
        metrics = trainer.train()

    console.print("\n[bold green]Training complete![/bold green]")
    console.print("Final metrics:")
    for key, value in metrics.items():
        console.print(f"  {key}: {value:.4f}")

    output_dir = Path(cfg.project.output_dir) / "checkpoints" / "final"
    console.print(f"\nModel saved to: [blue]{output_dir}[/blue]")


if __name__ == "__main__":
    main()
