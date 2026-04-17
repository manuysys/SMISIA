"""
AGRILION — Chatbot Service v2
================================
Improved orchestration: live context injection, response caching,
decision-escalation logic, and zero user-visible errors.
"""

import hashlib
import logging
import time
from typing import Optional
from dataclasses import dataclass, field

from llm_client import BaseLLMClient, FallbackLLMClient, create_llm_client, LLMConfig
from prompt_builder import PromptBuilder, build_system_context
from memory import ConversationMemory
from rule_based_engine import RuleBasedEngine, SiloState

logger = logging.getLogger(__name__)


# ─── Response ─────────────────────────────────────────────────────────────────

@dataclass
class ChatResponse:
    response: str
    session_id: str
    latency_ms: float
    context_used: dict = field(default_factory=dict)
    error: Optional[str] = None
    from_cache: bool = False
    fallback_layer: int = 0   # 0=LLM, 1=Ollama, 2=rules, 3=emergency

    def to_dict(self) -> dict:
        return {
            "response": self.response,
            "session_id": self.session_id,
            "latency_ms": round(self.latency_ms, 1),
            "context_used": self.context_used,
            "error": self.error,
            "from_cache": self.from_cache,
        }


# ─── Simple Response Cache ────────────────────────────────────────────────────

class _ResponseCache:
    """In-memory LRU-like cache for identical context+message pairs."""

    def __init__(self, max_size: int = 50, ttl_seconds: int = 120):
        self._cache: dict[str, tuple[str, float]] = {}
        self.max_size = max_size
        self.ttl = ttl_seconds

    def _key(self, message: str, context: dict) -> str:
        risk = context.get("risk_score", "")
        level = context.get("risk_level", "")
        silo = context.get("silo_id", "")
        raw = f"{silo}|{level}|{risk}|{message}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, message: str, context: dict) -> Optional[str]:
        k = self._key(message, context)
        if k in self._cache:
            text, ts = self._cache[k]
            if time.time() - ts < self.ttl:
                return text
            del self._cache[k]
        return None

    def set(self, message: str, context: dict, response: str):
        if len(self._cache) >= self.max_size:
            # Remove oldest entry
            oldest = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest]
        self._cache[self._key(message, context)] = (response, time.time())


# ─── Chatbot Service ──────────────────────────────────────────────────────────

class ChatbotService:
    """
    Main chatbot orchestrator for AGRILION.

    Flow:
        user message
            → fetch live silo context
            → check cache (skip LLM for repeated identical queries)
            → build prompt (context injected as readable text)
            → call LLM (3-layer fallback: OpenRouter → Ollama → rules)
            → update conversation memory
            → return ChatResponse
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        llm_config: Optional[LLMConfig] = None,
        max_memory_turns: int = 4,
        ai_service=None,
        risk_engine=None,
        enable_cache: bool = True,
    ):
        self.rule_engine = RuleBasedEngine()

        if llm_client:
            self.llm = llm_client
        else:
            try:
                self.llm = create_llm_client(llm_config, rule_engine=self.rule_engine)
            except Exception as e:
                logger.warning(f"LLM init failed ({e}), using rule-only mode")
                self.llm = _RuleOnlyClient(self.rule_engine)

        self.memory = ConversationMemory(max_turns=max_memory_turns)
        self.ai_service = ai_service
        self.risk_engine = risk_engine
        self._builders: dict[str, PromptBuilder] = {}
        self._cache = _ResponseCache() if enable_cache else None

    # ─── Public API ───────────────────────────────────────────────────────────

    def chat(
        self,
        message: str,
        silo_id: str = "SILO_001",
        session_id: str = "default",
        context_override: Optional[dict] = None,
        short_mode: bool = False,
    ) -> ChatResponse:
        """
        Process a user message and return a structured response.

        Args:
            message: user question
            silo_id: target silo identifier
            session_id: conversation session
            context_override: bypass live fetch (for testing)
            short_mode: request a concise ≤3-sentence response

        Returns:
            ChatResponse
        """
        t0 = time.time()
        context = context_override or self._fetch_context(silo_id)

        # Cache check (skip for high-risk states — always generate fresh)
        if self._cache and context.get("risk_score", 0) < 60:
            cached = self._cache.get(message, context)
            if cached:
                return ChatResponse(
                    response=cached,
                    session_id=session_id,
                    latency_ms=(time.time() - t0) * 1000,
                    context_used=context,
                    from_cache=True,
                )

        builder = self._get_builder(silo_id, context)
        history = self.memory.get_history(session_id)

        if short_mode:
            messages = builder.build_short_response_prompt(message)
        else:
            messages = builder.build_messages(message, conversation_history=history)

        try:
            if isinstance(self.llm, FallbackLLMClient):
                response_text = self.llm.complete(messages, context=context)
            else:
                response_text = self.llm.complete(messages)
        except Exception as e:
            logger.error(f"All LLM layers failed: {e}")
            response_text = self._emergency_response(context)

        self.memory.add(session_id, "user", message)
        self.memory.add(session_id, "assistant", response_text)

        if self._cache and response_text:
            self._cache.set(message, context, response_text)

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            latency_ms=(time.time() - t0) * 1000,
            context_used=context,
        )

    def summarize_silo(self, silo_id: str, session_id: str = "summary") -> ChatResponse:
        """Generate a natural-language silo status summary."""
        t0 = time.time()
        context = self._fetch_context(silo_id)
        builder = self._get_builder(silo_id, context)
        messages = builder.build_status_summary_prompt()
        try:
            if isinstance(self.llm, FallbackLLMClient):
                text = self.llm.complete(messages, context=context)
            else:
                text = self.llm.complete(messages)
        except Exception:
            _, state = RuleBasedEngine.from_context(context)
            text = self.rule_engine.respond(state, "resumen")
        return ChatResponse(
            response=text, session_id=session_id,
            latency_ms=(time.time() - t0) * 1000, context_used=context,
        )

    def explain_alert(self, alert: dict, silo_id: str = "SILO_001", session_id: str = "alert") -> ChatResponse:
        """Explain a specific alert."""
        t0 = time.time()
        context = self._fetch_context(silo_id)
        builder = self._get_builder(silo_id, context)
        messages = builder.build_alert_explanation_prompt(alert)
        try:
            if isinstance(self.llm, FallbackLLMClient):
                text = self.llm.complete(messages, context=context)
            else:
                text = self.llm.complete(messages)
        except Exception:
            lvl = alert.get("level", "")
            msg = alert.get("message", "")
            rec = alert.get("recommendation", "")
            text = f"Alerta [{lvl}]: {msg}\n\nAcción recomendada: {rec}"
        return ChatResponse(
            response=text, session_id=session_id,
            latency_ms=(time.time() - t0) * 1000, context_used=context,
        )

    def clear_session(self, session_id: str):
        self.memory.clear(session_id)

    # ─── Private ─────────────────────────────────────────────────────────────

    def _fetch_context(self, silo_id: str) -> dict:
        """Fetch live context from AIService, or return empty context."""
        if self.ai_service:
            try:
                result = self.ai_service.analyze_batch(silo_id)
                if result:
                    return build_system_context(
                        silo_id=silo_id,
                        sensor_values=None,
                        risk_score=result.risk_score,
                        risk_level=result.risk_level,
                        alerts=result.alerts,
                        predictions=result.predictions,
                        anomalies=result.anomalies,
                    )
            except Exception as e:
                logger.warning(f"Live context fetch failed for {silo_id}: {e}")
        return build_system_context(silo_id=silo_id)

    def _get_builder(self, silo_id: str, context: dict) -> PromptBuilder:
        if silo_id not in self._builders:
            self._builders[silo_id] = PromptBuilder(context)
        else:
            self._builders[silo_id].update_context(context)
        return self._builders[silo_id]

    def _emergency_response(self, context: dict) -> str:
        """Absolute last resort — uses rule engine directly."""
        try:
            _, state = RuleBasedEngine.from_context(context)
            return self.rule_engine.respond(state)
        except Exception:
            return (
                "Sistema temporalmente no disponible. "
                "Revise el panel de control para ver el estado actual del silo."
            )


# ─── Rule-only client (when no API key is set) ────────────────────────────────

class _RuleOnlyClient(BaseLLMClient):
    """Wraps rule engine as a drop-in LLM client."""

    def __init__(self, engine: RuleBasedEngine):
        super().__init__(LLMConfig())
        self._engine = engine

    def complete(self, messages: list[dict], context: dict = None, **kwargs) -> str:
        ctx = context or {}
        _, state = RuleBasedEngine.from_context(ctx)
        user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        return self._engine.respond(state, user_msg)
