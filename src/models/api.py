"""API-based model clients for Groq, HuggingFace, and Ollama."""

import logging
import os
from typing import Any

import httpx
from omegaconf import DictConfig

from .base import BaseModel, ModelFactory

logger = logging.getLogger(__name__)


@ModelFactory.register("api")
class APIModel(BaseModel):
    """Model accessed via OpenAI-compatible API (Groq, HuggingFace, etc.)."""

    def __init__(self, cfg: DictConfig):
        """Initialize API model.

        Args:
            cfg: Model configuration.
        """
        super().__init__(cfg)
        self.api_cfg = cfg.api
        self.provider = cfg.get("api_provider", "groq")
        self._client = None

    def load(self) -> None:
        """Initialize the API client."""
        if self._client is not None:
            return

        api_key = os.getenv(self.api_cfg.env_key)
        if not api_key:
            raise ValueError(
                f"API key not found. Set {self.api_cfg.env_key} environment variable."
            )

        if self.provider == "groq":
            self._init_groq_client(api_key)
        elif self.provider == "huggingface":
            self._init_huggingface_client(api_key)
        else:
            self._init_openai_compatible_client(api_key)

        logger.info(f"API client initialized for {self.provider}: {self.model_id}")

    def _init_groq_client(self, api_key: str) -> None:
        """Initialize Groq client."""
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
            self._client_type = "groq"
        except ImportError:
            logger.warning("groq package not installed, using httpx")
            self._init_openai_compatible_client(api_key)

    def _init_huggingface_client(self, api_key: str) -> None:
        """Initialize HuggingFace Inference client."""
        self._api_key = api_key
        self._base_url = self.api_cfg.base_url
        self._client = httpx.Client(
            timeout=self.api_cfg.get("timeout", 120),
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self._client_type = "huggingface"

    def _init_openai_compatible_client(self, api_key: str) -> None:
        """Initialize OpenAI-compatible client."""
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=api_key,
                base_url=self.api_cfg.base_url,
                timeout=self.api_cfg.get("timeout", 60),
            )
            self._client_type = "openai"
        except ImportError:
            raise ImportError("openai package required for API models")

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
            **kwargs: Additional parameters.

        Returns:
            Generated text.
        """
        if self._client is None:
            self.load()

        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, max_new_tokens, temperature, **kwargs)

    def chat(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a chat response.

        Args:
            messages: Chat messages.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional parameters.

        Returns:
            Assistant's response.
        """
        if self._client is None:
            self.load()

        gen_config = self.get_generation_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        if self._client_type == "huggingface":
            return self._chat_huggingface(messages, gen_config)
        elif self._client_type == "groq":
            return self._chat_groq(messages, gen_config)
        else:
            return self._chat_openai(messages, gen_config)

    def _chat_openai(
        self,
        messages: list[dict[str, str]],
        gen_config: dict[str, Any],
    ) -> str:
        """Chat using OpenAI-compatible API."""
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=gen_config["max_new_tokens"],
            temperature=gen_config["temperature"],
            top_p=gen_config.get("top_p", 0.9),
        )
        return response.choices[0].message.content

    def _chat_groq(
        self,
        messages: list[dict[str, str]],
        gen_config: dict[str, Any],
    ) -> str:
        """Chat using Groq API."""
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=gen_config["max_new_tokens"],
            temperature=gen_config["temperature"],
            top_p=gen_config.get("top_p", 0.9),
        )
        return response.choices[0].message.content

    def _chat_huggingface(
        self,
        messages: list[dict[str, str]],
        gen_config: dict[str, Any],
    ) -> str:
        """Chat using HuggingFace Inference API."""
        prompt = self._format_messages_for_hf(messages)

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": gen_config["max_new_tokens"],
                "temperature": gen_config["temperature"],
                "top_p": gen_config.get("top_p", 0.9),
                "return_full_text": False,
            },
        }

        url = f"{self._base_url}/{self.model_id}"
        response = self._client.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")
        return ""

    def _format_messages_for_hf(self, messages: list[dict[str, str]]) -> str:
        """Format messages for HuggingFace API."""
        formatted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted.append(f"<|system|>\n{content}</s>")
            elif role == "user":
                formatted.append(f"<|user|>\n{content}</s>")
            elif role == "assistant":
                formatted.append(f"<|assistant|>\n{content}</s>")
        formatted.append("<|assistant|>")
        return "\n".join(formatted)


@ModelFactory.register("ollama")
class OllamaModel(BaseModel):
    """Model accessed via Ollama local API."""

    def __init__(self, cfg: DictConfig):
        """Initialize Ollama model.

        Args:
            cfg: Model configuration.
        """
        super().__init__(cfg)
        self.api_cfg = cfg.get("api", {})
        self.base_url = self.api_cfg.get("base_url", "http://localhost:11434")
        self._client = None

    def load(self) -> None:
        """Initialize Ollama client."""
        if self._client is not None:
            return

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.api_cfg.get("timeout", 120),
        )

        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            if self.model_id not in models:
                logger.warning(
                    f"Model {self.model_id} not found locally. "
                    f"Run: ollama pull {self.model_id}"
                )
        except httpx.HTTPError as e:
            logger.warning(f"Could not connect to Ollama: {e}")

        logger.info(f"Ollama client initialized: {self.model_id}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text using Ollama.

        Args:
            prompt: Input prompt.
            max_new_tokens: Maximum tokens (num_predict in Ollama).
            temperature: Sampling temperature.
            **kwargs: Additional parameters.

        Returns:
            Generated text.
        """
        if self._client is None:
            self.load()

        gen_config = self.get_generation_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": gen_config["max_new_tokens"],
                "temperature": gen_config["temperature"],
                "top_p": gen_config.get("top_p", 0.9),
                "top_k": gen_config.get("top_k", 50),
            },
        }

        response = self._client.post("/api/generate", json=payload)
        response.raise_for_status()
        return response.json().get("response", "")

    def chat(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Chat using Ollama.

        Args:
            messages: Chat messages.
            max_new_tokens: Maximum tokens.
            temperature: Sampling temperature.
            **kwargs: Additional parameters.

        Returns:
            Assistant's response.
        """
        if self._client is None:
            self.load()

        gen_config = self.get_generation_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": gen_config["max_new_tokens"],
                "temperature": gen_config["temperature"],
                "top_p": gen_config.get("top_p", 0.9),
                "top_k": gen_config.get("top_k", 50),
            },
        }

        response = self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")

    def list_models(self) -> list[str]:
        """List available Ollama models.

        Returns:
            List of model names.
        """
        if self._client is None:
            self.load()

        response = self._client.get("/api/tags")
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]

    def pull_model(self, model_name: str | None = None) -> None:
        """Pull a model from Ollama registry.

        Args:
            model_name: Model to pull (defaults to configured model).
        """
        if self._client is None:
            self.load()

        model = model_name or self.model_id
        logger.info(f"Pulling model: {model}")

        response = self._client.post(
            "/api/pull",
            json={"name": model, "stream": False},
            timeout=600,
        )
        response.raise_for_status()
        logger.info(f"Model {model} pulled successfully")
