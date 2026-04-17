"""
AGRILION — LLM Client (PRO VERSION)
===================================
Robust client with:
- OpenRouter primary (LLaMA 70B free)
- Automatic fallback (Mistral free)
- Token optimization
- Error handling (404, 429, etc.)
"""

import os
import time
import logging
import httpx
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


# ─── Utils ────────────────────────────────────────────────────────────────────

def trim_messages(messages, max_chars=4000):
    """Reduce message size to avoid token overflow."""
    total = 0
    trimmed = []

    for m in reversed(messages):
        total += len(m["content"])
        if total > max_chars:
            break
        trimmed.insert(0, m)

    return trimmed


# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    provider: str = "openrouter"

    models: list = None  # 🔥 lista de modelos

    api_key: str = ""
    base_url: str = ""

    max_tokens: int = 300
    temperature: float = 0.4
    timeout: float = 8.0  # 🚀 Optimizado para evitar bloqueos largos
    silent: bool = False   # 🤫 Si es True, silencia logs no críticos

    max_retries: int = 1   # Reducido para fallar rápido y pasar al siguiente modelo
    retry_delay: float = 0.5

    @classmethod
    def from_env(cls):
        load_dotenv()
        return cls(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            silent=os.getenv("LLM_SILENT", "false").lower() == "true",
            # 🔥 MODELOS ESTABLES Y GRATUITOS (Task 4)
            models=[
                "meta-llama/llama-3.3-70b-instruct:free",
                "openchat/openchat-7b:free",
                "nousresearch/nous-hermes-2-mixtral:free",
                "gryphe/mythomist-7b:free"
            ]
        )


# ─── Base ─────────────────────────────────────────────────────────────────────

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
                return fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                if not self.config.silent:
                    logger.debug(f"LLM attempt {attempt + 1} failed: {e}")

                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay)
        raise last_err

class OllamaClient(BaseLLMClient):

    def complete(self, messages: list[dict], **kwargs) -> str:
        try:
            return self._call(messages)
        except Exception as e:
            if not self.config.silent:
                logger.debug(f"Ollama failed: {e}")
            raise e

    def _call(self, messages: list[dict]) -> str:

        prompt = "\n".join([m["content"] for m in messages])

        payload = {
            "model": "llama3",  # ⚠️ IMPORTANTE: debe coincidir con tu modelo instalado
            "prompt": prompt,
            "stream": False
        }

        with httpx.Client(timeout=10.0) as client:
            r = client.post(
                "http://localhost:11434/api/generate",
                json=payload
            )

            if r.status_code != 200:
                raise Exception(f"Ollama HTTP {r.status_code}")

            data = r.json()

            return data.get("response", "").strip()

# ─── OpenRouter Client ────────────────────────────────────────────────────────

class OpenRouterClient(BaseLLMClient):

    def complete(self, messages: list[dict], **kwargs) -> str:
        try:
            if not self.config.silent:
                logger.info("Falling back to Ollama...")

            ollama = OllamaClient(self.config)
            return ollama.complete(messages)

        except Exception:
            return (
                "⚠️ El asistente de IA no está disponible temporalmente. "
                "Por favor, revisa el panel de AGRILION."
            )

    def _call(self, messages: list[dict], **kwargs) -> str:

        messages = trim_messages(messages)

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        last_error = None

        for model in self.config.models:

            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            }

            try:
                with httpx.Client(timeout=self.config.timeout) as client:
                    r = client.post(
                        f"{self.config.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )

                    if r.status_code != 200:
                        if not self.config.silent:
                            logger.debug(f"Model {model} failed with HTTP {r.status_code}")
                        last_error = f"HTTP {r.status_code}"
                        continue

                    data = r.json()

                    # Safe parsing (Task 3)
                    choices = data.get("choices", [])
                    if not choices or not isinstance(choices, list):
                        if not self.config.silent:
                            logger.debug(f"Invalid format from {model}")
                        continue

                    content = choices[0].get("message", {}).get("content")
                    if not content:
                        if not self.config.silent:
                            logger.debug(f"Empty content from {model}")
                        continue

                    if not self.config.silent:
                        logger.info(f"SUCCESS with model: {model}")

                    return content.strip()

            except Exception as e:
                if not self.config.silent:
                    logger.debug(f"Exception with model {model}: {type(e).__name__}")
                last_error = e
                continue

        raise Exception("All models exhausted")


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_llm_client(config: Optional[LLMConfig] = None) -> BaseLLMClient:

    cfg = config or LLMConfig.from_env()

    if not cfg.api_key:
        logger.warning("No API key found → using fallback client")
        return FallbackClient(cfg)

    logger.info(f"LLM: OpenRouter | models={cfg.models}")
    return OpenRouterClient(cfg)


# ─── Fallback ─────────────────────────────────────────────────────────────────

class FallbackClient(BaseLLMClient):

    def complete(self, messages: list[dict], **kwargs) -> str:
        return (
            "⚠️ AI not configured. Set OPENROUTER_API_KEY in your environment.\n"
            "Get a free key at https://openrouter.ai"
        )