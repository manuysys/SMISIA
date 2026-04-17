"""
AGRILION — Prompt Builder
===========================
Builds structured prompts with injected system context for the LLM.
"""

import json
from typing import Optional
from datetime import datetime


# ─── System Role ─────────────────────────────────────────────────────────────

SYSTEM_ROLE = """You are AGRILION Assistant, an expert agricultural IoT monitoring assistant.
Your job is to help farmers and agronomists understand the state of their grain silo bags (silobolsas).

CAPABILITIES:
- Explain sensor readings (temperature, humidity, CO2)
- Interpret risk scores and alert levels
- Explain what anomalies mean for grain quality
- Guide users on recommended actions
- Answer questions about LSTM predictions and what they imply

RULES:
- NEVER invent sensor values or risk scores — use only the data provided in the context
- If data is missing, say it is not available
- Be concise and clear — farmers need actionable information
- Prefer simple language over technical jargon
- You may respond in Spanish or English depending on the user's language
- When risk is CRITICAL or WARNING, emphasize urgency
- Base your analysis strictly on the context provided
"""


# ─── Context Schema ───────────────────────────────────────────────────────────

def build_system_context(
    silo_id: str,
    sensor_values: Optional[dict] = None,
    risk_score: Optional[int] = None,
    risk_level: Optional[str] = None,
    alerts: Optional[list] = None,
    predictions: Optional[dict] = None,
    anomalies: Optional[dict] = None,
    last_updated: Optional[str] = None,
) -> dict:
    """
    Build a structured context dict to inject into the prompt.

    Args:
        silo_id: identifier of the silo
        sensor_values: {temperature, humidity, co2}
        risk_score: 0–100
        risk_level: NORMAL | WARNING | CRITICAL
        alerts: list of alert dicts
        predictions: next-step predictions
        anomalies: anomaly flags
        last_updated: ISO timestamp

    Returns:
        dict structured for prompt injection
    """
    return {
        "silo_id": silo_id,
        "timestamp": last_updated or datetime.now().isoformat(),
        "current_sensors": sensor_values or {},
        "risk_score": risk_score,
        "risk_level": risk_level or "UNKNOWN",
        "active_alerts": alerts or [],
        "predictions_next_step": predictions or {},
        "anomalies": anomalies or {},
    }


# ─── Prompt Builder ───────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Builds OpenAI-style message lists for the LLM including:
    - system role
    - injected AGRILION context
    - conversation history
    - current user message
    """

    def __init__(self, context: Optional[dict] = None):
        self.context = context or {}

    def update_context(self, context: dict):
        self.context = context

    def build_messages(
        self,
        user_message: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Build the full message list to send to the LLM.

        Args:
            user_message: the current user input
            conversation_history: list of prior {role, content} dicts

        Returns:
            list of {role, content} messages
        """
        messages = []

        # 1. System prompt
        system_content = SYSTEM_ROLE
        if self.context:
            system_content += f"\n\n--- CURRENT SYSTEM STATE ---\n{self._format_context()}"
        messages.append({"role": "system", "content": system_content})

        # 2. Conversation history (last N turns)
        if conversation_history:
            messages.extend(conversation_history)

        # 3. Current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    def _format_context(self) -> str:
        """Format context as readable JSON block."""
        ctx = self.context.copy()

        # Human-readable risk level with emoji
        level = ctx.get("risk_level", "UNKNOWN")
        emoji = {"NORMAL": "🟢", "WARNING": "🟡", "CRITICAL": "🔴"}.get(level, "⚪")
        ctx["risk_display"] = f"{emoji} {level} (score: {ctx.get('risk_score', 'N/A')}/100)"

        # Sensor summary
        sensors = ctx.get("current_sensors", {})
        if sensors:
            ctx["sensor_summary"] = {
                k: f"{v:.1f}" if isinstance(v, float) else v
                for k, v in sensors.items()
            }

        return json.dumps(ctx, indent=2, ensure_ascii=False, default=str)

    def build_status_summary_prompt(self) -> list[dict]:
        """Pre-built prompt: 'Summarize silo status'."""
        msg = (
            "Please provide a concise status summary of this silo. "
            "Include: overall condition, main concerns if any, and top recommended action."
        )
        return self.build_messages(msg)

    def build_alert_explanation_prompt(self, alert: dict) -> list[dict]:
        """Pre-built prompt: 'Explain this alert'."""
        msg = (
            f"Please explain this alert in simple terms and tell the farmer what to do:\n"
            f"Alert: {alert.get('category', 'unknown')} — {alert.get('message', '')}\n"
            f"Level: {alert.get('level', '')}"
        )
        return self.build_messages(msg)
