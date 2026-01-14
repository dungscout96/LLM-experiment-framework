"""Local model loading via HuggingFace Transformers."""

import logging
import platform
from typing import Any

import torch
from omegaconf import DictConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .base import BaseModel, ModelFactory

logger = logging.getLogger(__name__)


def get_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(dtype_str: str, device: str) -> torch.dtype:
    """Get PyTorch dtype from string.

    Args:
        dtype_str: Dtype string (float32, float16, bfloat16).
        device: Target device.

    Returns:
        PyTorch dtype.
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    dtype = dtype_map.get(dtype_str, torch.float16)

    if device == "mps" and dtype == torch.bfloat16:
        logger.warning("bfloat16 not supported on MPS, falling back to float16")
        dtype = torch.float16

    return dtype


@ModelFactory.register("local")
class LocalModel(BaseModel):
    """Local model loaded via HuggingFace Transformers."""

    def __init__(self, cfg: DictConfig):
        """Initialize local model.

        Args:
            cfg: Model configuration.
        """
        super().__init__(cfg)
        self.device = None
        self.dtype = None

    def load(self, device: str | None = None, dtype: str | None = None) -> None:
        """Load model from HuggingFace Hub.

        Args:
            device: Target device (auto, cpu, cuda, mps).
            dtype: Data type (float32, float16, bfloat16).
        """
        if self._model is not None:
            logger.info("Model already loaded")
            return

        self.device = device or get_device()
        dtype_str = dtype or self.cfg.get("dtype", "float16")
        self.dtype = get_dtype(dtype_str, self.device)

        logger.info(f"Loading {self.model_id} on {self.device} with {self.dtype}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            revision=self.cfg.get("revision", "main"),
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs = self._get_model_kwargs()

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            revision=self.cfg.get("revision", "main"),
            **model_kwargs,
        )

        if self.cfg.get("gradient_checkpointing", False):
            self._model.gradient_checkpointing_enable()

        logger.info(f"Model loaded: {self._model.config.name_or_path}")

    def _get_model_kwargs(self) -> dict[str, Any]:
        """Get model loading kwargs based on configuration."""
        kwargs = {
            "torch_dtype": self.dtype,
            "device_map": "auto" if self.device != "mps" else None,
        }

        quant_cfg = self.cfg.get("quantization", {})
        if quant_cfg.get("enabled", False) and self.device == "cuda":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quant_cfg.bits == 4,
                load_in_8bit=quant_cfg.bits == 8,
                bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=quant_cfg.get("double_quant", True),
            )
            logger.info(f"Using {quant_cfg.bits}-bit quantization")
        elif quant_cfg.get("enabled", False) and self.device == "mps":
            logger.warning(
                "bitsandbytes quantization not fully supported on MPS. "
                "Loading in full precision."
            )

        if self.cfg.get("use_flash_attention", False) and self.device == "cuda":
            kwargs["attn_implementation"] = "flash_attention_2"
        elif self.device == "mps":
            kwargs["attn_implementation"] = "eager"

        if self.device == "mps":
            kwargs.pop("device_map", None)

        return kwargs

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text (without prompt).
        """
        if self._model is None:
            self.load()

        gen_config = self.get_generation_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self.device == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        elif self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                pad_token_id=self._tokenizer.pad_token_id,
                **gen_config,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def chat(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a chat response.

        Args:
            messages: Chat messages with role and content.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional generation parameters.

        Returns:
            Assistant's response.
        """
        if self._model is None:
            self.load()

        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = self._format_messages_fallback(messages)

        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

    def _format_messages_fallback(self, messages: list[dict[str, str]]) -> str:
        """Format messages when chat template is not available."""
        formatted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        formatted.append("Assistant:")
        return "\n\n".join(formatted)

    def get_model_for_training(self):
        """Get model prepared for training.

        Returns:
            Model ready for PEFT/fine-tuning.
        """
        if self._model is None:
            self.load()
        return self._model

    def get_tokenizer(self):
        """Get tokenizer for training.

        Returns:
            Tokenizer instance.
        """
        if self._tokenizer is None:
            self.load()
        return self._tokenizer
