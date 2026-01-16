"""Local model loading via HuggingFace Transformers."""

import logging
import os
import platform
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
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


def get_hf_token(cfg: DictConfig) -> str | None:
    """Get HuggingFace token from config or environment.

    Args:
        cfg: Model configuration.

    Returns:
        HuggingFace token or None.
    """
    # Check config first
    token = cfg.get("hf_token", None)
    if token and token != "null":
        return token

    # Fall back to environment variables
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return token


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
        self.hf_token = get_hf_token(cfg)
        self.is_multimodal = cfg.get("multimodal", False)
        self._processor = None  # For multimodal models

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

        if self.is_multimodal:
            self._load_multimodal_model()
        else:
            self._load_text_model()

        if self.cfg.get("gradient_checkpointing", False):
            self._model.gradient_checkpointing_enable()

        logger.info(f"Model loaded: {self._model.config.name_or_path}")

    def _load_text_model(self) -> None:
        """Load a text-only model."""
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            revision=self.cfg.get("revision", "main"),
            token=self.hf_token,
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs = self._get_model_kwargs()

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            revision=self.cfg.get("revision", "main"),
            token=self.hf_token,
            **model_kwargs,
        )

    def _load_multimodal_model(self) -> None:
        """Load a multimodal vision-language model."""
        logger.info("Loading multimodal model with processor")

        # Load processor for handling both text and images
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            revision=self.cfg.get("revision", "main"),
            token=self.hf_token,
        )

        # Set tokenizer reference for compatibility
        self._tokenizer = self._processor.tokenizer
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs = self._get_model_kwargs()

        # Load the vision-language model
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            revision=self.cfg.get("revision", "main"),
            token=self.hf_token,
            **model_kwargs,
        )

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

    def get_processor(self):
        """Get processor for multimodal models.

        Returns:
            Processor instance (for multimodal) or tokenizer (for text-only).
        """
        if self._model is None:
            self.load()
        return self._processor if self._processor else self._tokenizer

    def generate_with_image(
        self,
        prompt: str,
        image: str | Image.Image,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text from a prompt with an image.

        Args:
            prompt: Input prompt/question about the image.
            image: PIL Image or path to image file.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text response.

        Raises:
            ValueError: If model is not multimodal.
        """
        if not self.is_multimodal:
            raise ValueError(
                f"Model {self.model_id} is not multimodal. "
                "Use generate() for text-only models."
            )

        if self._model is None:
            self.load()

        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        gen_config = self.get_generation_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

        # Process inputs using the multimodal processor
        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )

        # Move to device
        if self.device == "mps":
            inputs = {k: v.to("mps") if hasattr(v, "to") else v for k, v in inputs.items()}
        elif self.device == "cuda":
            inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                pad_token_id=self._tokenizer.pad_token_id,
                **gen_config,
            )

        # Decode output, excluding input tokens
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def chat_with_images(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a chat response with image support.

        Args:
            messages: Chat messages with role and content. Content can be:
                - A string (text only)
                - A list of dicts with "type" (text/image) and content
                  Example: [{"type": "image", "image": <PIL.Image or path>},
                           {"type": "text", "text": "What is this?"}]
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional generation parameters.

        Returns:
            Assistant's response.

        Raises:
            ValueError: If model is not multimodal.
        """
        if not self.is_multimodal:
            raise ValueError(
                f"Model {self.model_id} is not multimodal. "
                "Use chat() for text-only models."
            )

        if self._model is None:
            self.load()

        # Extract images from messages
        images = []
        processed_messages = []

        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")

            if isinstance(content, list):
                # Multi-part content with images
                text_parts = []
                for part in content:
                    if part.get("type") == "image":
                        img = part.get("image")
                        if isinstance(img, str):
                            img = Image.open(img).convert("RGB")
                        images.append(img)
                        text_parts.append("<image>")
                    elif part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                processed_messages.append({
                    "role": role,
                    "content": " ".join(text_parts),
                })
            else:
                processed_messages.append({"role": role, "content": content})

        # Apply chat template
        if hasattr(self._processor, "apply_chat_template"):
            prompt = self._processor.apply_chat_template(
                processed_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        elif hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                processed_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = self._format_messages_fallback(processed_messages)

        gen_config = self.get_generation_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

        # Process with images if present
        if images:
            inputs = self._processor(
                text=prompt,
                images=images if len(images) > 1 else images[0],
                return_tensors="pt",
            )
        else:
            inputs = self._processor(
                text=prompt,
                return_tensors="pt",
            )

        # Move to device
        if self.device == "mps":
            inputs = {k: v.to("mps") if hasattr(v, "to") else v for k, v in inputs.items()}
        elif self.device == "cuda":
            inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                pad_token_id=self._tokenizer.pad_token_id,
                **gen_config,
            )

        # Decode output
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)
