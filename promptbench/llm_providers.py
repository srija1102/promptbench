"""
llm_providers.py — Abstract LLM provider interface with retry logic.

Classes:
    LLMProvider  — Abstract base class
    GroqProvider — Groq API (OpenAI-compatible)
    OpenAIProvider — OpenAI API
    AnthropicProvider — Anthropic Messages API

All providers include:
  - Retry logic (max 3 attempts, exponential backoff)
  - Timeout handling
  - API key validation
  - Cost estimation (token approximation)
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    MAX_RETRIES = 3
    BASE_BACKOFF = 1.0  # seconds; doubles each retry

    def __init__(self, api_key: str, timeout: int = 60) -> None:
        """
        Args:
            api_key: Provider-specific API key.
            timeout: HTTP request timeout in seconds.
        """
        if not api_key or not api_key.strip():
            raise ValueError(
                f"{self.__class__.__name__}: api_key must not be empty.\n"
                "Set the appropriate environment variable for your provider."
            )
        self._api_key = api_key.strip()
        self._timeout = timeout

    @abstractmethod
    def generate(self, prompt: str, model: str) -> str:
        """
        Send *prompt* to *model* and return the text response.

        Args:
            prompt: The full prompt string.
            model:  Provider-specific model identifier.

        Returns:
            The model's text response.

        Raises:
            RuntimeError: On unrecoverable API errors after all retries.
        """

    def _with_retry(self, fn) -> str:
        """
        Execute *fn* with exponential backoff retries.

        Args:
            fn: A zero-argument callable that may raise an exception.

        Returns:
            The return value of *fn* on success.

        Raises:
            RuntimeError: After MAX_RETRIES failed attempts.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return fn()
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                wait = self.BASE_BACKOFF * (2 ** (attempt - 1))
                logger.warning(
                    "Attempt %d/%d failed (%s). Retrying in %.1fs...",
                    attempt, self.MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
            except requests.HTTPError as exc:
                # 429 = rate limit → retry; 4xx client errors → no retry
                if exc.response is not None and exc.response.status_code == 429:
                    last_exc = exc
                    wait = self.BASE_BACKOFF * (2 ** (attempt - 1))
                    logger.warning(
                        "Rate limited. Attempt %d/%d. Retrying in %.1fs...",
                        attempt, self.MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(str(exc)) from exc
        raise RuntimeError(
            f"LLM call failed after {self.MAX_RETRIES} attempts: {last_exc}"
        )

    @staticmethod
    def estimate_cost(prompt: str, response: str, cost_per_1k_tokens: float) -> float:
        """
        Approximate cost based on character-count token estimation.

        Uses the rough heuristic of ~4 chars per token.

        Args:
            prompt:             The prompt string sent to the model.
            response:           The response string received.
            cost_per_1k_tokens: Cost in USD per 1,000 tokens.

        Returns:
            Estimated cost in USD.
        """
        approx_tokens = (len(prompt) + len(response)) / 4
        return (approx_tokens / 1000) * cost_per_1k_tokens


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------

class GroqProvider(LLMProvider):
    """Groq cloud inference API (OpenAI-compatible endpoint)."""

    _BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
    _COST_PER_1K = 0.0001  # approximate

    @classmethod
    def from_env(cls) -> "GroqProvider":
        """Create a GroqProvider using GROQ_API_KEY from the environment."""
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set.\n"
                "Get a free key at https://console.groq.com and run:\n"
                "  export GROQ_API_KEY=<your-key>"
            )
        return cls(api_key=key)

    def generate(self, prompt: str, model: str) -> str:
        def _call():
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
            resp = requests.post(
                self._BASE_URL, json=payload, headers=headers, timeout=self._timeout
            )
            if not resp.ok:
                exc = requests.HTTPError(response=resp)
                exc.response = resp
                raise exc
            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                raise RuntimeError(f"Unexpected Groq response format: {data}") from e

        return self._with_retry(_call)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """OpenAI chat completions API."""

    _BASE_URL = "https://api.openai.com/v1/chat/completions"
    _COST_PER_1K = 0.002  # approximate (gpt-3.5 pricing baseline)

    @classmethod
    def from_env(cls) -> "OpenAIProvider":
        """Create an OpenAIProvider using OPENAI_API_KEY from the environment."""
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set.\n"
                "Set it with: export OPENAI_API_KEY=<your-key>"
            )
        return cls(api_key=key)

    def generate(self, prompt: str, model: str) -> str:
        def _call():
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
            resp = requests.post(
                self._BASE_URL, json=payload, headers=headers, timeout=self._timeout
            )
            resp.raise_for_status()
            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                raise RuntimeError(f"Unexpected OpenAI response format: {data}") from e

        return self._with_retry(_call)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """Anthropic Messages API."""

    _BASE_URL = "https://api.anthropic.com/v1/messages"
    _COST_PER_1K = 0.003  # approximate (claude-haiku baseline)

    @classmethod
    def from_env(cls) -> "AnthropicProvider":
        """Create an AnthropicProvider using ANTHROPIC_API_KEY from the environment."""
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set.\n"
                "Set it with: export ANTHROPIC_API_KEY=<your-key>"
            )
        return cls(api_key=key)

    def generate(self, prompt: str, model: str) -> str:
        def _call():
            headers = {
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": prompt}],
            }
            resp = requests.post(
                self._BASE_URL, json=payload, headers=headers, timeout=self._timeout
            )
            resp.raise_for_status()
            data = resp.json()
            try:
                return data["content"][0]["text"]
            except (KeyError, IndexError) as e:
                raise RuntimeError(f"Unexpected Anthropic response format: {data}") from e

        return self._with_retry(_call)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_provider(provider_name: str, api_key: str) -> LLMProvider:
    """
    Instantiate the correct LLMProvider for *provider_name*.

    Args:
        provider_name: One of 'groq', 'openai', 'anthropic'.
        api_key:       Provider API key.

    Returns:
        LLMProvider instance.

    Raises:
        ValueError: If provider_name is not recognised.
    """
    mapping = {
        "groq": GroqProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }
    cls = mapping.get(provider_name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Valid options: {', '.join(sorted(mapping))}"
        )
    return cls(api_key=api_key)
