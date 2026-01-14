#!/usr/bin/env python3
"""Evaluation script for trained models."""

import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from peft import PeftModel
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import LLMDataset
from src.models import LocalModel

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Set up logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def load_trained_model(
    base_model_id: str,
    checkpoint_path: str | Path,
    device: str = "auto",
) -> tuple:
    """Load a trained model from checkpoint.

    Args:
        base_model_id: HuggingFace model ID for base model.
        checkpoint_path: Path to saved checkpoint.
        device: Target device.

    Returns:
        Tuple of (model, tokenizer).
    """
    checkpoint_path = Path(checkpoint_path)

    console.print(f"Loading base model: [blue]{base_model_id}[/blue]")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map=device if device != "mps" else None,
        trust_remote_code=True,
    )

    adapter_config = checkpoint_path / "adapter_config.json"
    if adapter_config.exists():
        console.print(f"Loading PEFT adapter from: [blue]{checkpoint_path}[/blue]")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
    else:
        console.print(f"Loading full model from: [blue]{checkpoint_path}[/blue]")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float16,
            device_map=device if device != "mps" else None,
        )

    if device == "mps":
        model = model.to("mps")

    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    device: str = "auto",
) -> str:
    """Generate a response from the model.

    Args:
        model: The model.
        tokenizer: The tokenizer.
        prompt: Input prompt.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        device: Target device.

    Returns:
        Generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt")

    if device == "mps":
        inputs = {k: v.to("mps") for k, v in inputs.items()}
        # MPS has numerical precision issues with sampling, use greedy decoding
        temperature = 0.0
    elif device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def interactive_eval(model, tokenizer, device: str = "auto") -> None:
    """Run interactive evaluation session.

    Args:
        model: The model.
        tokenizer: The tokenizer.
        device: Target device.
    """
    console.print("\n[bold green]Interactive Evaluation Mode[/bold green]")
    console.print("Type 'quit' to exit, 'clear' to clear history\n")

    history = []

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            history = []
            console.print("[dim]History cleared[/dim]")
            continue

        if hasattr(tokenizer, "apply_chat_template"):
            messages = history + [{"role": "user", "content": user_input}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"User: {user_input}\nAssistant:"

        with console.status("[bold green]Generating..."):
            response = generate_response(
                model,
                tokenizer,
                prompt,
                device=device,
            )

        console.print(f"[bold green]Assistant:[/bold green] {response}\n")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run evaluation with Hydra configuration.

    Args:
        cfg: Hydra configuration.
    """
    setup_logging(cfg.get("verbose", False))

    console.print("\n[bold blue]LLM Evaluation[/bold blue]")

    checkpoint_path = cfg.get("checkpoint_path")
    if not checkpoint_path:
        checkpoint_path = Path(cfg.project.output_dir) / "checkpoints" / "final"

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint_path}[/red]")
        console.print("Train a model first or specify checkpoint_path")
        return

    device = cfg.hardware.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    console.print(f"Device: [green]{device}[/green]")

    model, tokenizer = load_trained_model(
        base_model_id=cfg.model.model_id,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    mode = cfg.get("eval_mode", "interactive")

    if mode == "interactive":
        interactive_eval(model, tokenizer, device=device)

    elif mode == "dataset":
        logger = logging.getLogger(__name__)

        dataset = LLMDataset(cfg.data)
        dataset.load()

        test_data = dataset.dataset.get("test") or dataset.dataset.get("validation")
        if test_data is None:
            console.print("[red]No test/validation data available[/red]")
            return

        table = Table(title="Evaluation Results")
        table.add_column("Prompt", style="cyan", max_width=50)
        table.add_column("Expected", style="green", max_width=50)
        table.add_column("Generated", style="yellow", max_width=50)

        num_samples = min(cfg.get("num_eval_samples", 10), len(test_data))

        for i in range(num_samples):
            sample = test_data[i]
            prompt = sample.get(cfg.data.columns.instruction, "")
            expected = sample.get(cfg.data.columns.output, "")

            generated = generate_response(
                model,
                tokenizer,
                prompt,
                device=device,
            )

            table.add_row(
                prompt[:50] + "..." if len(prompt) > 50 else prompt,
                expected[:50] + "..." if len(expected) > 50 else expected,
                generated[:50] + "..." if len(generated) > 50 else generated,
            )

        console.print(table)

    console.print("\n[bold green]Evaluation complete![/bold green]")


if __name__ == "__main__":
    main()
