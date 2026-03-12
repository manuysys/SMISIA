"""
SMISIA — Chatbot Backend
Formatea respuestas naturales en español a partir de datos de inferencia.
"""
import re
from typing import Optional
from src.api.schemas import ChatResponse


# Emojis por estado
STATUS_EMOJI = {
    "bien": "✅",
    "tolerable": "⚠️",
    "problema": "🚨",
    "critico": "🔴",
}

# Plantillas de respuesta
TEMPLATES = {
    "status": {
        "brief": (
            "{emoji} Silobolsa {silo_id} — {status_upper} "
            "(confianza {confidence:.0%}). {summary} "
            "¿Querés ver la tendencia 7d?"
        ),
        "detail": (
            "**Silobolsa {silo_id}** — Estado: **{status_upper}**\n\n"
            "| Métrica | Valor |\n"
            "|---------|-------|\n"
            "{metrics_table}"
            "\n**Tendencias (24h):**\n{trends}\n\n"
            "**Top drivers:**\n{drivers}\n\n"
            "**Recomendación:** {recommendation}"
        ),
    },
    "prediction": {
        "brief": (
            "El modelo predice {prob:.0%} probabilidad de empeoramiento a "
            "{horizon} días. Driver principal: {driver}."
        ),
    },
    "no_data": {
        "brief": (
            "No tengo datos recientes para {identifier}. "
            "Ejecutá una inferencia con /infer primero."
        ),
    },
    "unknown": {
        "brief": (
            "No entendí la consulta. Podés preguntar:\n"
            "• ¿Cuál es el estado de la silobolsa X?\n"
            "• ¿Va a empeorar en 3 días?\n"
            "• Mostrame la tendencia del silo X"
        ),
    },
}


def format_chat_response(
    message: str,
    silo_id: Optional[str] = None,
    cached_data: Optional[dict] = None,
) -> ChatResponse:
    """
    Transforma datos de inferencia en respuesta natural en español.

    Args:
        message: Mensaje del usuario
        silo_id: ID del silo (extraído del mensaje o proporcionado)
        cached_data: Datos cacheados de la última inferencia
    """
    message_lower = message.lower().strip()

    # Detectar intención
    if any(kw in message_lower for kw in ["estado", "cómo está", "como esta", "status"]):
        return _format_status(silo_id, cached_data)

    elif any(kw in message_lower for kw in ["empeorar", "predicción", "prediccion", "futuro", "va a"]):
        return _format_prediction(message_lower, silo_id, cached_data)

    elif any(kw in message_lower for kw in ["tendencia", "trend", "historial"]):
        return _format_status(silo_id, cached_data)

    elif silo_id and cached_data:
        return _format_status(silo_id, cached_data)

    else:
        return ChatResponse(
            brief=TEMPLATES["unknown"]["brief"],
            silo_id=silo_id,
        )


def _format_status(silo_id: Optional[str], data: Optional[dict]) -> ChatResponse:
    """Formatea respuesta de estado."""
    if not data or not silo_id:
        identifier = silo_id or "ese silo"
        return ChatResponse(
            brief=TEMPLATES["no_data"]["brief"].format(identifier=identifier),
            silo_id=silo_id,
        )

    status = data.get("status", "desconocido")
    emoji = STATUS_EMOJI.get(status, "❓")
    confidence = data.get("confidence", 0)

    # Brief
    brief = TEMPLATES["status"]["brief"].format(
        emoji=emoji,
        silo_id=silo_id,
        status_upper=status.upper(),
        confidence=confidence,
        summary=data.get("summary", ""),
    )

    # Detail — tabla de métricas
    metrics = data.get("metrics", {})
    metrics_table = ""
    for name, info in metrics.items():
        if isinstance(info, dict):
            val = info.get("value", "N/A")
            unit = info.get("unit", "")
            metrics_table += f"| {name} | {val} {unit} |\n"

    # Tendencias
    trends = ""
    trend_data = data.get("trend", {})
    for key, val in trend_data.items():
        direction = "↑" if val > 0 else "↓" if val < 0 else "→"
        trends += f"  • {key}: {direction} {abs(val):.2f}\n"
    if not trends:
        trends = "  Sin datos de tendencia disponibles.\n"

    # Drivers
    drivers = ""
    explanations = data.get("explanations", [])
    for i, exp in enumerate(explanations, 1):
        feat = exp.get("feature", "N/A") if isinstance(exp, dict) else exp.feature
        impact = exp.get("impact", 0) if isinstance(exp, dict) else exp.impact
        drivers += f"  {i}. {feat}: {impact:.0%}\n"
    if not drivers:
        drivers = "  Sin datos de explicabilidad disponibles.\n"

    detail = TEMPLATES["status"]["detail"].format(
        silo_id=silo_id,
        status_upper=status.upper(),
        metrics_table=metrics_table or "| Sin datos | - |\n",
        trends=trends,
        drivers=drivers,
        recommendation=data.get("recommended_action", "N/A"),
    )

    return ChatResponse(brief=brief, detail=detail, silo_id=silo_id)


def _format_prediction(
    message: str,
    silo_id: Optional[str],
    data: Optional[dict],
) -> ChatResponse:
    """Formatea respuesta de predicción."""
    if not data or not silo_id:
        identifier = silo_id or "ese silo"
        return ChatResponse(
            brief=TEMPLATES["no_data"]["brief"].format(identifier=identifier),
            silo_id=silo_id,
        )

    # Extraer horizonte del mensaje
    horizon = 3  # default
    match = re.search(r"(\d+)\s*d[ií]as?", message)
    if match:
        horizon = int(match.group(1))

    # Usar raw_scores como proxy de probabilidad de empeoramiento
    raw = data.get("raw_scores", {})
    prob_worsen = raw.get("problema", 0) + raw.get("critico", 0)

    # Driver principal
    explanations = data.get("explanations", [])
    if explanations:
        top = explanations[0]
        driver_name = top.get("feature", "N/A") if isinstance(top, dict) else top.feature
        driver = f"subida continua de {driver_name}"
    else:
        driver = "múltiples factores combinados"

    brief = TEMPLATES["prediction"]["brief"].format(
        prob=prob_worsen,
        horizon=horizon,
        driver=driver,
    )

    return ChatResponse(brief=brief, silo_id=silo_id)


# ---------------------------------------------------------------------------
# Ejemplos de diálogo (few-shot) — documentados en el código
# ---------------------------------------------------------------------------
DIALOG_EXAMPLES = [
    {
        "user": "¿Cuál es el estado de la silobolsa A12?",
        "bot": (
            "🚨 Silobolsa A12 — PROBLEMA (confianza 78%). "
            "Humedad en aumento; revisar ventilación. "
            "¿Querés ver la tendencia 7d?"
        ),
    },
    {
        "user": "¿Va a empeorar en 3 días?",
        "bot": (
            "El modelo predice 65% probabilidad de empeoramiento a 3 días. "
            "Driver principal: subida continua de humedad "
            "(delta 72h = +4.2 p.p.)."
        ),
    },
    {
        "user": "Mostrame la tendencia del silo B05",
        "bot": (
            "✅ Silobolsa B05 — BIEN (confianza 94%). "
            "Condiciones estables dentro de parámetros normales. "
            "¿Querés ver la tendencia 7d?"
        ),
    },
]
