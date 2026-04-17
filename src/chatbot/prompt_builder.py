"""
AGRILION — Prompt Builder v2
==============================
Clean context injection, strict Spanish-only behavior, domain-aware formatting.
No raw dicts dumped into prompts. No language mixing.
"""

from typing import Optional
from datetime import datetime


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres el Asistente de AGRILION, sistema experto en monitoreo de silobolsas agrícolas.

REGLAS ABSOLUTAS:
1. Responde SIEMPRE en español. Sin excepciones.
2. Usa ÚNICAMENTE los datos del contexto del sistema. Nunca inventes valores.
3. Si no tienes datos, dilo explícitamente: "No hay datos disponibles para este silo."
4. Sé conciso, claro y orientado a decisiones.
5. Evita frases genéricas o de relleno.

DOMINIO:
- Monitoreas silobolsas con sensores de temperatura (°C), humedad (%) y CO₂ (ppm)
- Umbrales críticos: temperatura > 30°C, humedad > 70%, CO₂ > 700 ppm
- Los hongos se desarrollan con humedad > 70% + temperatura > 28°C
- CO₂ elevado indica actividad biológica (fermentación, respiración fúngica)
- El deterioro del grano es irreversible si las condiciones críticas persisten > 48h

FORMATO DE RESPUESTA (cuando hay datos de sensores):
Estado: [NIVEL] [EMOJI]
Silo: [ID]

Problema:
[Descripción clara]

Riesgo:
[Explicación técnica simple]

Acciones recomendadas:
• [Acción 1]
• [Acción 2]
• [Acción 3]

Predicción: (solo si aplica)
[Tendencia esperada]

URGENCIA:
- Si risk_score > 60: enfatiza urgencia
- Si risk_score > 80: indica CRÍTICO, recomienda acción inmediata"""


# ─── Context Formatter ────────────────────────────────────────────────────────

def format_context_block(context: dict) -> str:
    """
    Convert context dict into a clean, human-readable block for the prompt.
    Avoids raw JSON dumps — the LLM receives structured text, not code.
    """
    if not context:
        return "Sin datos de sistema disponibles."

    silo = context.get("silo_id", "—")
    risk_score = context.get("risk_score")
    risk_level = context.get("risk_level", "DESCONOCIDO")
    timestamp = context.get("timestamp", "")
    sensors = context.get("current_sensors", {})
    alerts = context.get("active_alerts", [])
    predictions = context.get("predictions_next_step", {})
    anomalies = context.get("anomalies", {})

    # Risk display
    emoji_map = {"NORMAL": "🟢", "WARNING": "⚠️", "CRITICAL": "🔴"}
    emoji = emoji_map.get(risk_level, "⚪")
    score_str = f"{risk_score}/100" if risk_score is not None else "N/D"

    lines = [
        "=== ESTADO ACTUAL DEL SISTEMA ===",
        f"Silo: {silo}",
        f"Nivel de riesgo: {risk_level} {emoji} (score: {score_str})",
    ]

    if timestamp:
        try:
            ts = timestamp[:19].replace("T", " ")
        except Exception:
            ts = str(timestamp)
        lines.append(f"Última actualización: {ts}")

    # Sensors
    if sensors:
        temp = sensors.get("temperature")
        hum = sensors.get("humidity")
        co2 = sensors.get("co2")
        parts = []
        if temp is not None:
            parts.append(f"Temperatura: {float(temp):.1f}°C")
        if hum is not None:
            parts.append(f"Humedad: {float(hum):.1f}%")
        if co2 is not None:
            parts.append(f"CO₂: {float(co2):.0f} ppm")
        if parts:
            lines.append("Sensores: " + " | ".join(parts))

    # Alerts
    if alerts:
        lines.append(f"Alertas activas: {len(alerts)}")
        for a in alerts[:3]:
            lvl = a.get("level", "?")
            cat = a.get("category", "?")
            msg = a.get("message", "")
            lines.append(f"  [{lvl}] {cat}: {msg}")
    else:
        lines.append("Alertas activas: ninguna")

    # Predictions
    if predictions:
        pred_parts = []
        for k, v in predictions.items():
            if v is not None:
                unit = "°C" if "temp" in k else "%" if "hum" in k else " ppm"
                pred_parts.append(f"{k}: {float(v):.1f}{unit}")
        if pred_parts:
            lines.append("Predicción próxima lectura: " + " | ".join(pred_parts))

    # Anomalies
    if anomalies.get("is_anomaly"):
        affected = anomalies.get("affected_sensors", [])
        lines.append(f"Anomalía detectada en: {', '.join(affected) if affected else 'sistema'}")

    lines.append("=================================")
    return "\n".join(lines)


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
    """Build the canonical context dict (unchanged API for backward compat)."""
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
    Builds structured message lists for the LLM.

    Changes vs v1:
    - Context injected as readable text block (not raw JSON)
    - Stricter system prompt (Spanish-only, no hallucinations)
    - History trimmed to avoid token overflow
    - Pre-built prompts for common operations
    """

    MAX_HISTORY_MESSAGES = 6  # 3 turns max to avoid token overflow

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
        Build message list for LLM call.

        Args:
            user_message: current user input
            conversation_history: prior {role, content} pairs

        Returns:
            list of {role, content} messages, token-safe
        """
        messages = []

        # System prompt + context block
        context_block = format_context_block(self.context)
        system_content = f"{SYSTEM_PROMPT}\n\n{context_block}"
        messages.append({"role": "system", "content": system_content})

        # Trimmed history (keep last N messages)
        if conversation_history:
            trimmed = conversation_history[-self.MAX_HISTORY_MESSAGES:]
            messages.extend(trimmed)

        messages.append({"role": "user", "content": user_message})
        return messages

    def build_status_summary_prompt(self) -> list[dict]:
        """Pre-built: silo status summary."""
        return self.build_messages(
            "Genera un resumen del estado actual de este silo. "
            "Indica el estado general, los problemas detectados y la acción más urgente."
        )

    def build_alert_explanation_prompt(self, alert: dict) -> list[dict]:
        """Pre-built: explain a specific alert."""
        lvl = alert.get("level", "")
        cat = alert.get("category", "")
        msg = alert.get("message", "")
        return self.build_messages(
            f"Explica esta alerta al agricultor en términos simples y dile qué debe hacer:\n"
            f"Tipo: {cat} | Nivel: {lvl}\n"
            f"Mensaje: {msg}"
        )

    def build_short_response_prompt(self, user_message: str) -> list[dict]:
        """Pre-built: concise one-paragraph answer for UI cards."""
        messages = self.build_messages(user_message)
        # Append instruction for brevity
        messages[-1]["content"] += "\n\n(Responde en máximo 3 oraciones.)"
        return messages
