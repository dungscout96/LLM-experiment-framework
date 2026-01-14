"""Base model interface for LLM experiments."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all LLM model wrappers."""

    def __init__(self, cfg: DictConfig):
        """Initialize model with configuration.

        Args:
            cfg: Model configuration from Hydra.
        """
        self.cfg = cfg
        self.model_id = cfg.model_id
        self.name = cfg.name
        self._model = None
        self._tokenizer = None

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text.
        """
        pass

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a chat response.

        Args:
            messages: List of chat messages with role and content.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional generation parameters.

        Returns:
            Assistant's response.
        """
        pass

    def get_generation_config(self, **overrides: Any) -> dict[str, Any]:
        """Get generation configuration with optional overrides.

        Args:
            **overrides: Parameters to override defaults.

        Returns:
            Generation configuration dictionary.
        """
        gen_cfg = self.cfg.generation
        config = {
            "max_new_tokens": gen_cfg.max_new_tokens,
            "temperature": gen_cfg.temperature,
            "top_p": gen_cfg.get("top_p", 0.9),
            "top_k": gen_cfg.get("top_k", 50),
            "do_sample": gen_cfg.get("do_sample", True),
            "repetition_penalty": gen_cfg.get("repetition_penalty", 1.0),
        }
        config.update({k: v for k, v in overrides.items() if v is not None})
        return config

    @property
    def model(self):
        """Get the underlying model object."""
        return self._model

    @property
    def tokenizer(self):
        """Get the tokenizer."""
        return self._tokenizer

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, model_id={self.model_id})"


class ModelFactory:
    """Factory for creating model instances."""

    _registry: dict[str, type[BaseModel]] = {}

    @classmethod
    def register(cls, source: str):
        """Decorator to register a model class.

        Args:
            source: Model source identifier (local, api, ollama).

        Returns:
            Decorator function.
        """
        def decorator(model_cls: type[BaseModel]):
            cls._registry[source] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, cfg: DictConfig) -> BaseModel:
        """Create a model instance from configuration.

        Args:
            cfg: Model configuration.

        Returns:
            Model instance.

        Raises:
            ValueError: If model source is not registered.
        """
        source = cfg.source
        if source not in cls._registry:
            raise ValueError(
                f"Unknown model source: {source}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[source](cfg)

    @classmethod
    def available_sources(cls) -> list[str]:
        """Get list of available model sources."""
        return list(cls._registry.keys())
