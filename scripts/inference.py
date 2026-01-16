#!/usr/bin/env python3
"""Unified inference script supporting all model sources.

Supports all model sources via Hydra config:
- Local HuggingFace models: model=medgemma_4b, model=llama3_8b
- API models (Groq, HF Inference): model=api_groq, model=api_huggingface
- Ollama: model=ollama

Examples:
    # MedGemma (local) with prompt
    python scripts/inference.py model=medgemma_27b_text +prompt="What is diabetes?"

    # Groq API
    python scripts/inference.py model=api_groq +prompt="Hello!"

    # Ollama
    python scripts/inference.py model=ollama +prompt="Explain machine learning"

    # Interactive mode (no prompt)
    python scripts/inference.py model=medgemma_4b

    # With image (multimodal models)
    python scripts/inference.py model=medgemma_4b +image=path/to/xray.jpg +prompt="Describe"

    # Override generation settings
    python scripts/inference.py model=api_groq +prompt="Hello" +max_tokens=100 +temperature=0.5
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

# Load environment variables
load_dotenv()


def run_inference(cfg: DictConfig):
    """Run inference with the configured model.

    Args:
        cfg: Hydra configuration.
    """
    # Import here to avoid slow startup for --help
    from src.models.base import ModelFactory

    # Ensure model classes are registered
    from src.models import local, api  # noqa: F401

    print(f"Creating model: {cfg.model.name} (source: {cfg.model.source})")

    # Create model via factory
    model = ModelFactory.create(cfg.model)

    # Load the model
    print("Loading model...")
    if hasattr(model, 'load'):
        if cfg.model.source == "local":
            # Use hardware.device from config or auto
            device = cfg.get("device", cfg.hardware.get("device", "auto"))
            if device == "auto":
                device = None
            model.load(device=device)
        else:
            model.load()

    print(f"Model ready: {model}")

    # Get inference parameters from config
    prompt = cfg.get("prompt", None)
    image_path = cfg.get("image", None)
    max_tokens = cfg.get("max_tokens", 256)
    temperature = cfg.get("temperature", None)
    use_chat = cfg.get("use_chat", True)

    # Check if multimodal
    is_multimodal = cfg.model.get("multimodal", False)

    if prompt:
        # Single prompt mode
        if image_path and is_multimodal:
            print(f"\nProcessing image: {image_path}")
            response = model.generate_with_image(
                prompt=prompt,
                image=image_path,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
        elif use_chat:
            messages = [{"role": "user", "content": prompt}]
            response = model.chat(
                messages=messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            response = model.generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
        print(f"\nResponse: {response}")
    else:
        # Interactive mode
        run_interactive(model, max_tokens, temperature, is_multimodal)


def run_interactive(model, max_tokens: int, temperature: float | None, is_multimodal: bool):
    """Run interactive chat mode.

    Args:
        model: The loaded model.
        max_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        is_multimodal: Whether model supports images.
    """
    print("\n" + "=" * 50)
    print("Interactive mode. Commands:")
    print("  quit     - Exit")
    if is_multimodal:
        print("  /image   - Set image path for next message")
        print("  /clear   - Clear image")
    print("=" * 50 + "\n")

    current_image = None

    while True:
        try:
            prompt = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not prompt:
            continue

        if prompt.lower() == "quit":
            break

        if prompt.startswith("/image "):
            if is_multimodal:
                current_image = prompt[7:].strip()
                print(f"Image set: {current_image}")
            else:
                print("This model doesn't support images.")
            continue

        if prompt == "/clear":
            current_image = None
            print("Image cleared.")
            continue

        try:
            if current_image and is_multimodal:
                response = model.generate_with_image(
                    prompt=prompt,
                    image=current_image,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                current_image = None  # Clear after use
            else:
                messages = [{"role": "user", "content": prompt}]
                response = model.chat(
                    messages=messages,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")

    print("Goodbye!")


def list_models():
    """List available model configurations."""
    print("\nAvailable models (use model=<name>):\n")
    model_dir = project_root / "configs" / "model"
    for f in sorted(model_dir.glob("*.yaml")):
        try:
            cfg = OmegaConf.load(f)
            source = cfg.get("source", "unknown")
            model_id = cfg.get("model_id", "N/A")
            multimodal = cfg.get("multimodal", False)
            mm_tag = " [multimodal]" if multimodal else ""
            print(f"  {f.stem:20s} {source:8s} {model_id}{mm_tag}")
        except Exception:
            print(f"  {f.stem:20s} (error loading config)")
    print()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point with Hydra config.

    Usage:
        # Run with prompt
        python scripts/inference.py model=api_groq +prompt="Hello!"

        # Interactive mode
        python scripts/inference.py model=medgemma_4b

        # List models
        python scripts/inference.py +list_models=true
    """
    # Check for list_models flag
    if cfg.get("list_models", False):
        list_models()
        return

    run_inference(cfg)


if __name__ == "__main__":
    main()
