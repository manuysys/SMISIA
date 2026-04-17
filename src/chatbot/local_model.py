import httpx
import logging

logger = logging.getLogger(__name__)


class LocalLLMClient:
    """
    Cliente local usando Ollama
    Siempre responde (fallback final)
    """

    def __init__(self, model: str = "llama3"):
        self.base_url = "http://localhost:11434"
        self.model = model
        self.timeout = 5.0

    def complete(self, messages: list[dict], **kwargs) -> str:
        try:
            prompt = self._messages_to_prompt(messages)

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }

            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(f"{self.base_url}/api/generate", json=payload)

                if r.status_code != 200:
                    return self._safe_fallback()

                data = r.json()
                return data.get("response", "").strip() or self._safe_fallback()

        except Exception as e:
            logger.debug(f"Ollama error: {e}")
            return self._safe_fallback()

    def _messages_to_prompt(self, messages):
        """Convierte mensajes estilo OpenAI a prompt simple"""
        parts = []

        for m in messages:
            role = m["role"]
            content = m["content"]

            if role == "system":
                parts.append(f"[SYSTEM]\n{content}")
            elif role == "user":
                parts.append(f"[USER]\n{content}")
            elif role == "assistant":
                parts.append(f"[ASSISTANT]\n{content}")

        parts.append("[ASSISTANT]\n")
        return "\n\n".join(parts)

    def _safe_fallback(self):
        return (
            "⚠️ El sistema está en modo offline. "
            "No se pudo generar una respuesta completa, pero los datos siguen disponibles."
        )