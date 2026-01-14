#!/usr/bin/env python3
"""Simple inference script for trained models."""

import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(
    base_model_id: str,
    adapter_path: str,
    device: str = "auto",
):
    """Load base model with LoRA adapter.

    Args:
        base_model_id: HuggingFace model ID.
        adapter_path: Path to saved LoRA adapter.
        device: Target device (auto, cuda, mps, cpu).

    Returns:
        Tuple of (model, tokenizer).
    """
    print(f"Loading tokenizer from {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Loading base model: {base_model_id} on {device}")

    # Load base model
    model_kwargs = {
        "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
        "trust_remote_code": True,
    }

    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    elif device == "mps":
        model_kwargs["attn_implementation"] = "eager"

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)

    # Load adapter
    print(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    if device == "mps":
        model = model.to("mps")

    model.eval()
    return model, tokenizer, device


def generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 256,
    use_chat_template: bool = True,
):
    """Generate response from the model.

    Args:
        model: The model.
        tokenizer: The tokenizer.
        prompt: User prompt.
        device: Device the model is on.
        max_new_tokens: Maximum tokens to generate.
        use_chat_template: Whether to apply chat template.

    Returns:
        Generated response.
    """
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt

    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    if device in ["cuda", "mps"]:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for stability
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on trained model")
    parser.add_argument("--base-model", type=str, required=True, help="Base model ID")
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/mps/cpu)")
    parser.add_argument("--prompt", type=str, help="Single prompt to run")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens")
    parser.add_argument("--no-chat-template", action="store_true", help="Don't use chat template")
    args = parser.parse_args()

    # Load model
    model, tokenizer, device = load_model(args.base_model, args.adapter, args.device)

    if args.prompt:
        # Single prompt mode
        response = generate(
            model, tokenizer, args.prompt, device,
            max_new_tokens=args.max_tokens,
            use_chat_template=not args.no_chat_template,
        )
        print(f"\nResponse: {response}")
    else:
        # Interactive mode
        print("\nInteractive mode. Type 'quit' to exit.\n")
        while True:
            try:
                prompt = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if prompt.lower() == "quit":
                break

            if not prompt:
                continue

            response = generate(
                model, tokenizer, prompt, device,
                max_new_tokens=args.max_tokens,
                use_chat_template=not args.no_chat_template,
            )
            print(f"Assistant: {response}\n")

    print("Goodbye!")


if __name__ == "__main__":
    main()
