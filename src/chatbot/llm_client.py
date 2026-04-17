"""
AGRILION — LLM Client v3 (PRO)
================================
3-layer fallback:
    1. OpenRouter (multi-model)
    2. Ollama (local)
    3. Rule-based engine

Optimizado para producción:
- Multi-model fallback real
- Manejo inteligente de errores (no retry en 404/401)
- UTF-8 clean output
- Logging claro
"""

import os
import time
import unicodedata
import logging
import httpx
from abc import ABC, abstractmethod
from typing import Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    provider: str = "openrouter"

    # 🔥 MULTI-MODEL (orden = prioridad)
    models: List[str] = field(default_factory=lambda: [
        "meta-llama/llama-3.1-8b-instruct",
        "mistralai/mistral-7b-instruct",
        "google/gemma-7b-it"
    ])

    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"

    max_tokens: int = 500
    temperature: float = 0.3
    timeout: float = 10.0

    max_retries: int = 1
    retry_delay: float = 0.5

    # Ollama fallback
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3"),
        )


# ─── Sanitizer ───────────────────────────────────────────────────────────────

def sanitize(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufffd", "").replace("\x00", "")

    replacements = {
        "Â°": "°",
        "Ã¡": "á",
        "Ã©": "é",
        "Ã³": "ó",
        "Ãº": "ú",
        "Ã±": "ñ",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text.strip()


# ─── Base ────────────────────────────────────────────────────────────────────

class BaseLLMClient(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def complete(self, messages: list[dict], **kwargs) -> str:
        pass

    def _retry(self, fn, *args, **kwargs):
        last_err = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = fn(*args, **kwargs)
                return sanitize(result)

            except httpx.HTTPStatusError as e:
                status = e.response.status_code

                # ❌ NO retry en errores de cliente
                if status in (401, 403, 404):
                    raise

                last_err = e

            except Exception as e:
                last_err = e

            if attempt < self.config.max_retries:
                time.sleep(self.config.retry_delay)

        raise last_err


# ─── OpenRouter (MULTI-MODEL) ────────────────────────────────────────────────

class OpenRouterClient(BaseLLMClient):

    def complete(self, messages: list[dict], **kwargs) -> str:
        return self._retry(self._call, messages, **kwargs)

    def _call(self, messages: list[dict], **kwargs) -> str:
        if not self.config.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        last_error = None

        for model in self.config.models:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
            }

            try:
                with httpx.Client(timeout=self.config.timeout) as client:
                    r = client.post(
                        f"{self.config.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )

                    r.raise_for_status()

                    content = r.json()["choices"][0]["message"]["content"]

                    logger.info(f"[LLM] SUCCESS → {model}")
                    return content

            except Exception as e:
                logger.warning(f"[LLM] Model failed → {model}: {e}")
                last_error = e
                continue

        raise last_error


# ─── Ollama ──────────────────────────────────────────────────────────────────

class OllamaClient(BaseLLMClient):

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

    def _to_prompt(self, messages):
        parts = []

        for m in messages:
            if m["role"] == "system":
                parts.append(f"[SYSTEM]\n{m['content']}")
            elif m["role"] == "user":
                parts.append(f"[USER]\n{m['content']}")
            elif m["role"] == "assistant":
                parts.append(f"[ASSISTANT]\n{m['content']}")

        return "\n\n".join(parts)


# ─── Fallback Client ─────────────────────────────────────────────────────────

class FallbackLLMClient(BaseLLMClient):

    def __init__(self, config: LLMConfig, rule_engine=None):
        super().__init__(config)
        self.primary = OpenRouterClient(config)
        self.ollama = OllamaClient(config)
        self.rule_engine = rule_engine

    def complete(self, messages: list[dict], context: dict = None, **kwargs) -> str:

        # ── LAYER 1: OpenRouter ──
        try:
            return self.primary.complete(messages, **kwargs)
        except Exception as e:
            logger.warning(f"[Fallback] OpenRouter failed: {e}")

        # ── LAYER 2: Ollama ──
        try:
            return self.ollama.complete(messages, **kwargs)
        except Exception as e:
            logger.warning(f"[Fallback] Ollama failed: {e}")

        # ── LAYER 3: Rule Engine ──
        if self.rule_engine and context:
            logger.info("[Fallback] Using Rule Engine")

            from rule_based_engine import RuleBasedEngine

            engine, state = RuleBasedEngine.from_context(context)

            user_msg = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"),
                ""
            )

            return engine.respond(state, user_msg)

        return "Sistema temporalmente no disponible."


# ─── Factory ─────────────────────────────────────────────────────────────────

def create_llm_client(config: Optional[LLMConfig] = None, rule_engine=None):
    cfg = config or LLMConfig.from_env()

    logger.info(f"LLM INIT → models={cfg.models}")

    return FallbackLLMClient(cfg, rule_engine=rule_engine)


# ─── Compatibilidad ──────────────────────────────────────────────────────────

class FallbackClient(FallbackLLMClient):
    pass