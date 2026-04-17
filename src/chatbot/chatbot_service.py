"""
AGRILION — Chatbot Service
============================
Core chatbot orchestrator: fetches context → builds prompt → calls LLM → returns response.
"""

import logging
import time
from typing import Optional
from dataclasses import dataclass, field

from .llm_client import BaseLLMClient, FallbackClient, create_llm_client, LLMConfig
from .prompt_builder import PromptBuilder, build_system_context
from .memory import ConversationMemory

logger = logging.getLogger(__name__)


# ─── Response Schema ──────────────────────────────────────────────────────────

@dataclass
class ChatResponse:
    response: str
    session_id: str
    latency_ms: float
    context_used: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "response": self.response,
            "session_id": self.session_id,
            "latency_ms": round(self.latency_ms, 1),
            "context_used": self.context_used,
            "error": self.error,
        }


# ─── Chatbot Service ──────────────────────────────────────────────────────────

class ChatbotService:
    """
    Main chatbot service for AGRILION.

    Integrates with the AI pipeline to inject real system state
    into every LLM call.

    Usage:
        svc = ChatbotService()
        response = svc.chat("Is my silo safe?", silo_id="SILO_001", session_id="abc")
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        llm_config: Optional[LLMConfig] = None,
        max_memory_turns: int = 5,
        ai_service=None,   # optional: AIService instance for real-time context
        risk_engine=None,  # optional: for computing live risk
    ):
        # LLM client
        if llm_client:
            self.llm = llm_client
        else:
            try:
                self.llm = create_llm_client(llm_config)
            except Exception as e:
                logger.warning(f"LLM init failed ({e}), using fallback")
                self.llm = FallbackClient(LLMConfig())

        self.memory = ConversationMemory(max_turns=max_memory_turns)
        self.ai_service = ai_service
        self.risk_engine = risk_engine
        self._prompt_builders: dict[str, PromptBuilder] = {}  # per silo

    def chat(
        self,
        message: str,
        silo_id: str = "SILO_001",
        session_id: str = "default",
        context_override: Optional[dict] = None,
    ) -> ChatResponse:
        """
        Main entry point: process a user message and return a response.

        Args:
            message: user's question or command
            silo_id: which silo to load context for
            session_id: conversation session identifier
            context_override: manually injected context (bypasses live fetch)

        Returns:
            ChatResponse with LLM response and metadata
        """
        t_start = time.time()

        # 1. Fetch system context
        context = context_override or self._fetch_context(silo_id)

        # 2. Get or create prompt builder for this silo
        builder = self._get_builder(silo_id, context)

        # 3. Get conversation history for this session
        history = self.memory.get_history(session_id)

        # 4. Build messages
        messages = builder.build_messages(message, conversation_history=history)

        # 5. Call LLM
        try:
            response_text = self.llm.complete(messages)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            response_text = self._degraded_response(message, context)
            error = str(e)
        else:
            error = None

        # 6. Update memory
        self.memory.add(session_id, "user", message)
        self.memory.add(session_id, "assistant", response_text)

        latency = (time.time() - t_start) * 1000
        logger.info(f"[{silo_id}] Chat session={session_id} latency={latency:.0f}ms")

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            latency_ms=latency,
            context_used=context,
            error=error,
        )

    def summarize_silo(self, silo_id: str, session_id: str = "summary") -> ChatResponse:
        """Convenience: generate a silo status summary."""
        context = self._fetch_context(silo_id)
        builder = self._get_builder(silo_id, context)
        messages = builder.build_status_summary_prompt()
        try:
            text = self.llm.complete(messages)
        except Exception as e:
            text = self._degraded_response("status summary", context)
        return ChatResponse(
            response=text,
            session_id=session_id,
            latency_ms=0,
            context_used=context,
        )

    def explain_alert(
        self, alert: dict, silo_id: str = "SILO_001", session_id: str = "alert"
    ) -> ChatResponse:
        """Convenience: explain a specific alert."""
        context = self._fetch_context(silo_id)
        builder = self._get_builder(silo_id, context)
        messages = builder.build_alert_explanation_prompt(alert)
        try:
            text = self.llm.complete(messages)
        except Exception as e:
            text = f"Alert: {alert.get('message', '')} — Level: {alert.get('level', '')}"
        return ChatResponse(
            response=text,
            session_id=session_id,
            latency_ms=0,
            context_used=context,
        )

    def clear_session(self, session_id: str):
        """Clear conversation history for a session."""
        self.memory.clear(session_id)

    # ─── Private ─────────────────────────────────────────────────────────────

    def _fetch_context(self, silo_id: str) -> dict:
        """Fetch real-time system state from AIService if available."""
        if self.ai_service:
            try:
                result = self.ai_service.analyze_batch(silo_id)
                if result:
                    return build_system_context(
                        silo_id=silo_id,
                        sensor_values=result.anomalies.get("sensor_values"),
                        risk_score=result.risk_score,
                        risk_level=result.risk_level,
                        alerts=result.alerts,
                        predictions=result.predictions,
                        anomalies=result.anomalies,
                    )
            except Exception as e:
                logger.warning(f"Could not fetch live context for {silo_id}: {e}")

        # No live data — return minimal context
        return build_system_context(silo_id=silo_id)

    def _get_builder(self, silo_id: str, context: dict) -> PromptBuilder:
        if silo_id not in self._prompt_builders:
            self._prompt_builders[silo_id] = PromptBuilder(context)
        else:
            self._prompt_builders[silo_id].update_context(context)
        return self._prompt_builders[silo_id]

    def _degraded_response(self, message: str, context: dict) -> str:
        """Graceful fallback when LLM is unavailable."""
        risk_level = context.get("risk_level", "UNKNOWN")
        risk_score = context.get("risk_score", "N/A")
        sensors = context.get("current_sensors", {})
        alerts = context.get("active_alerts", [])

        lines = [
            "⚠️ AI assistant temporarily unavailable. Here is the raw system data:",
            f"Silo: {context.get('silo_id', 'N/A')}",
            f"Risk: {risk_level} (score {risk_score}/100)",
        ]
        if sensors:
            lines.append(f"Sensors: {sensors}")
        if alerts:
            lines.append(f"Active alerts: {len(alerts)}")
            for a in alerts[:2]:
                lines.append(f"  - [{a.get('level')}] {a.get('message', '')}")
        return "\n".join(lines)
