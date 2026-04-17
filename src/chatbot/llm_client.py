"""
AGRILION — LLM Client v2
==========================
3-layer fallback: OpenRouter → Ollama (local) → Rule-Based Engine.
Handles retries, timeouts, encoding safety, and graceful degradation.
"""

import os
import time
import unicodedata
import logging
import httpx
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    provider: str = "openrouter"
    model: str = "mistralai/mistral-7b-instruct:free"
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 600
    temperature: float = 0.3        # lower = more consistent/factual
    timeout: float = 12.0
    max_retries: int = 1
    retry_delay: float = 0.5
    # Ollama fallback
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = os.getenv("LLM_PROVIDER", "openrouter")
        mapping = {
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "model": os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free"),
                "api_key": os.getenv("OPENROUTER_API_KEY", ""),
            },
            "huggingface": {
                "base_url": "https://api-inference.huggingface.co/models",
                "model": os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"),
                "api_key": os.getenv("HUGGINGFACE_API_KEY", ""),
            },
            "generic": {
                "base_url": os.getenv("LLM_BASE_URL", ""),
                "model": os.getenv("LLM_MODEL", ""),
                "api_key": os.getenv("LLM_API_KEY", ""),
            },
        }
        overrides = mapping.get(provider, mapping["openrouter"])
        return cls(
            provider=provider,
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3"),
            **overrides,
        )


# ─── Output Sanitization ──────────────────────────────────────────────────────

def sanitize(text: str) -> str:
    """
    Ensure output is clean UTF-8 with proper agricultural symbols.
    Removes replacement characters and normalizes encoding.
    """
    if not text:
        return ""
    # Normalize unicode (handles accents, special chars)
    text = unicodedata.normalize("NFC", text)
    # Remove replacement characters (U+FFFD) and null bytes
    text = text.replace("\ufffd", "").replace("\x00", "")
    # Normalize degree symbol variants
    text = text.replace("Â°", "°").replace("â€¦", "…").replace("Ã©", "é")
    return text.strip()


# ─── Abstract Base ────────────────────────────────────────────────────────────

class BaseLLMClient(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def complete(self, messages: list[dict], **kwargs) -> str:
        pass

    def _retry(self, fn, *args, **kwargs) -> str:
        last_err = None
        for attempt in range(self.config.max_retries + 1):
            try:
                result = fn(*args, **kwargs)
                return sanitize(result)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                # Don't retry client errors except rate limits
                if status in (401, 403, 404) or (400 <= status < 429):
                    raise
                last_err = e
            except Exception as e:
                last_err = e

            if attempt < self.config.max_retries:
                wait = self.config.retry_delay * (2 ** attempt)
                logger.warning(f"LLM attempt {attempt + 1} failed ({last_err}), retrying in {wait:.1f}s")
                time.sleep(wait)

        raise last_err


# ─── OpenRouter ───────────────────────────────────────────────────────────────

class OpenRouterClient(BaseLLMClient):
    """
    Primary LLM provider. Free tier: https://openrouter.ai
    Free models: mistralai/mistral-7b-instruct:free, meta-llama/llama-3-8b-instruct:free
    """

    def complete(self, messages: list[dict], **kwargs) -> str:
        return self._retry(self._call, messages, **kwargs)

    def _call(self, messages: list[dict], **kwargs) -> str:
        if not self.config.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://agrilion.io",
            "X-Title": "AGRILION Assistant",
        }
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        with httpx.Client(timeout=self.config.timeout) as client:
            r = client.post(f"{self.config.base_url}/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]


# ─── Ollama (local fallback) ──────────────────────────────────────────────────

class OllamaClient(BaseLLMClient):
    """
    Secondary fallback — local Ollama instance.
    Install: https://ollama.com | Models: ollama pull llama3
    """

    def complete(self, messages: list[dict], **kwargs) -> str:
        return self._retry(self._call, messages, **kwargs)

    def _call(self, messages: list[dict], **kwargs) -> str:
        prompt = self._to_prompt(messages)
        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
            },
        }
        with httpx.Client(timeout=self.config.timeout) as client:
            r = client.post(f"{self.config.ollama_url}/api/generate", json=payload)
            r.raise_for_status()
            return r.json().get("response", "")

    @staticmethod
    def _to_prompt(messages: list[dict]) -> str:
        """Convert OpenAI-style messages to a single prompt string."""
        parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                parts.append(f"<s>[INST] {content} [/INST]")
            elif role == "user":
                parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                parts.append(content)
        return "\n".join(parts)


# ─── HuggingFace ─────────────────────────────────────────────────────────────

class HuggingFaceClient(BaseLLMClient):
    """Free HuggingFace Inference API — slower but available."""

    def complete(self, messages: list[dict], **kwargs) -> str:
        return self._retry(self._call, messages, **kwargs)

    def _call(self, messages: list[dict], **kwargs) -> str:
        if not self.config.api_key:
            raise ValueError("HUGGINGFACE_API_KEY not set")
        prompt = OllamaClient._to_prompt(messages)  # same format works
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "return_full_text": False,
            },
        }
        url = f"{self.config.base_url}/{self.config.model}"
        with httpx.Client(timeout=self.config.timeout) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return data[0].get("generated_text", "") if isinstance(data, list) else str(data)


# ─── 3-Layer Fallback Client ──────────────────────────────────────────────────

class FallbackLLMClient(BaseLLMClient):
    """
    Orchestrates 3-layer fallback:
      1. Primary (OpenRouter or configured provider)
      2. Ollama (local)
      3. Rule-based engine (ALWAYS succeeds)

    Users never see internal errors.
    """

    def __init__(self, config: LLMConfig, rule_engine=None):
        super().__init__(config)
        self.primary = _build_primary(config)
        self.ollama = OllamaClient(config)
        self.rule_engine = rule_engine  # injected by ChatbotService

    def complete(self, messages: list[dict], context: dict = None, **kwargs) -> str:
        # Layer 1: Primary LLM
        try:
            result = self.primary.complete(messages, **kwargs)
            logger.debug("LLM layer 1 (primary) succeeded")
            return result
        except Exception as e1:
            logger.warning(f"Primary LLM failed: {e1}")

        # Layer 2: Ollama
        try:
            result = self.ollama.complete(messages, **kwargs)
            logger.info("LLM layer 2 (Ollama) succeeded")
            return result
        except Exception as e2:
            logger.warning(f"Ollama failed: {e2}")

        # Layer 3: Rule-based (ALWAYS succeeds)
        logger.info("LLM layer 3 (rule-based) activated")
        if self.rule_engine and context:
            from rule_based_engine import RuleBasedEngine, SiloState
            engine, state = RuleBasedEngine.from_context(context)
            # Extract user message from messages
            user_msg = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
            )
            return engine.respond(state, user_msg)

        return (
            "Sistema de respaldo activado. Verifique el estado del silo "
            "directamente en el panel de control."
        )


def _build_primary(config: LLMConfig) -> BaseLLMClient:
    """Build the primary client based on config."""
    clients = {
        "openrouter": OpenRouterClient,
        "huggingface": HuggingFaceClient,
        "generic": OpenRouterClient,  # generic uses same OpenAI-compat API
    }
    cls = clients.get(config.provider, OpenRouterClient)
    return cls(config)


def create_llm_client(config: Optional[LLMConfig] = None, rule_engine=None) -> FallbackLLMClient:
    """Factory: create a FallbackLLMClient from config or environment."""
    cfg = config or LLMConfig.from_env()
    logger.info(f"LLM client: {cfg.provider} / {cfg.model}")
    return FallbackLLMClient(cfg, rule_engine=rule_engine)


# ─── Legacy alias ─────────────────────────────────────────────────────────────
# Keep backward compatibility with any code that used FallbackClient
class FallbackClient(FallbackLLMClient):
    pass
