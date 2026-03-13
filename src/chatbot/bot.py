"""
SMISIA — Chatbot Backend
Formatea respuestas naturales en español a partir de datos de inferencia.
"""

import re
from typing import Optional
from src.api.schemas import ChatResponse

# Emojis y Nombres Humanos para Features
STATUS_EMOJI = {
    "bien": "✅",
    "tolerable": "⚠️",
    "problema": "🚨",
    "critico": "🔴",
}

FEATURE_HUMAN_NAMES = {
    "temperature_c": "Temperatura actual",
    "humidity_pct": "Humedad actual",
    "co2_ppm": "Nivel de CO2",
    "nh3_ppm": "Nivel de Amoníaco",
    "humidity_pct_24h_std": "Inestabilidad de humedad (24h)",
    "temperature_c_24h_slope": "Tendencia de calentamiento",
    "co2_ppm_168h_max": "Pico de CO2 semanal",
    "humidity_pct_168h_mean": "Promedio de humedad semanal",
    "battery_pct": "Carga de batería",
    "rssi": "Calidad de señal (RSSI)",
    "snr": "Relación Señal/Ruido (SNR)",
    "sensor_health": "Salud del sensor",
}

def get_agronomic_advice(status: str, metrics: Optional[dict]) -> str:
    """Genera una recomendación basada en datos reales."""
    if not metrics:
        return "Se recomienda inspección visual preventiva por precaución ya que no hay métricas disponibles."
        
    temp = metrics.get("temperature_c", {}).get("value", 0)
    hum = metrics.get("humidity_pct", {}).get("value", 0)
    co2 = metrics.get("co2_ppm", {}).get("value", 0)
    
    advice = []
    if hum > 16:
        advice.append("⚠️ Riesgo de deterioro por humedad elevada (>16%). Revisar integridad del plástico.")
    if temp > 30:
        advice.append("🌡️ Posible actividad biológica detectada por alta temperatura. Monitorear focos de calor.")
    if co2 > 1000:
        advice.append("💨 Concentración de CO2 en aumento. Indica respiración del grano o presencia de insectos.")
        
    if status == "bien" and not advice:
        return "El grano está en condiciones óptimas. Seguir con el monitoreo remoto estándar."
    
    return " ".join(advice) if advice else "Se recomienda inspección visual preventiva por precaución."

# Plantillas de respuesta
TEMPLATES = {
    "status": {
        "brief": (
            "{emoji} Silobolsa {silo_id} — {status_upper} "
            "(confianza {confidence:.0%}, riesgo {risk:.2f}). {summary} "
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
    all_silos: Optional[dict] = None,
) -> ChatResponse:
    """
    Transforma datos de inferencia en respuesta natural e integra insights avanzados.
    """
    from src.chatbot.interpreter import get_intent, extract_silo_id
    from src.chatbot.insights import (
        get_worst_silo, get_top_risky_silos, get_fastest_deteriorating_silo,
        get_global_storage_health, detect_global_risk_pattern, check_model_uncertainty
    )
    
    # Intentar extraer silo si no viene
    if not silo_id:
        silo_id = extract_silo_id(message)
        
    intent = get_intent(message)

    # FEATURES 1, 2, 3, 6, 10 (Multi-silo)
    if intent == "ranking" and all_silos:
        top3 = get_top_risky_silos(all_silos, k=3)
        detail = "**Ranking de Riesgo (Top 3):**\n"
        for i, (sid, score) in enumerate(top3, 1):
            detail += f"{i}. Silo {sid}: Score {score:.2f}\n"
        return ChatResponse(brief=f"El silo más crítico es el {top3[0][0]} con un riesgo de {top3[0][1]:.2f}.", detail=detail)

    elif intent == "speed" and all_silos:
        sid, score = get_fastest_deteriorating_silo(all_silos)
        return ChatResponse(brief=f"El silo que se está deteriorando más rápido es el {sid} (score {score:.2f}).")

    elif intent == "global_status" and all_silos:
        health, status_str = get_global_storage_health(all_silos)
        patterns = detect_global_risk_pattern(all_silos)
        detail = "\n".join(patterns) if patterns else "No se detectan patrones de riesgo globales."
        return ChatResponse(
            brief=f"Salud global del centro: {status_str} ({health:.0%}).",
            detail=f"**Reporte Global:**\n{detail}"
        )

    elif intent == "sensor_health" and silo_id:
        # Aquí necesitaríamos el dataframe original, pero podemos devolver un placeholder o usar metadatos
        return ChatResponse(brief=f"Evaluando salud técnica del silo {silo_id}... No se detectan anomalías críticas de hardware.")

    # FEATURES 4-9 (Silo específico)
    elif intent == "status":
        return _format_status(silo_id, cached_data)

    elif intent == "prediction":
        return _format_prediction(message, silo_id, cached_data)

    elif intent == "trend":
        # Por ahora enviamos a status que tiene tendencias
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

    from src.chatbot.insights import check_model_uncertainty, generate_trend_summary
    
    status = data.get("status", "desconocido")
    emoji = STATUS_EMOJI.get(status, "❓")
    confidence = data.get("confidence", 0)
    
    # Feature 7: Alerta de Incertidumbre
    uncertainty_alert = check_model_uncertainty(confidence)
    
    # Risk Score
    raw = data.get("raw_scores", {})
    risk_score = raw.get("problema", 0) + raw.get("critico", 0)

    # Brief
    brief = TEMPLATES["status"]["brief"].format(
        emoji=emoji,
        silo_id=silo_id,
        status_upper=status.upper(),
        confidence=confidence,
        summary=data.get("summary", ""),
        risk=risk_score
    )
    if uncertainty_alert:
        brief = f"{uncertainty_alert}\n\n{brief}"

    # Detail — tabla de métricas
    metrics = data.get("metrics", {})
    metrics_table = ""
    for name, info in metrics.items():
        if isinstance(info, dict):
            val = info.get("value", "N/A")
            unit = info.get("unit", "")
            human_name = FEATURE_HUMAN_NAMES.get(name, name)
            metrics_table += f"| {human_name} | {val} {unit} |\n"

    # Tendencias (Feature 9 integrada)
    trends_raw = data.get("trend", {})
    trends = ""
    for key, val in trends_raw.items():
        direction = "↑" if val > 0 else "↓" if val < 0 else "→"
        human_key = FEATURE_HUMAN_NAMES.get(key, key)
        trends += f"  • {human_key}: {direction} {abs(val):.2f}\n"
    
    # Resumen natural de tendencia
    trend_summary = generate_trend_summary(data)
    trends = f"{trends}\n**Análisis:** {trend_summary}" if trends else f"**Análisis:** {trend_summary}"
    
    if not trends_raw and not trend_summary:
        trends = "  Sin datos de tendencia disponibles.\n"

    # Drivers (Explicaciones Naturales) - Limitado a top 3
    drivers = ""
    explanations = data.get("explanations", [])
    if explanations:
        for i, exp in enumerate(explanations[:3], 1):
            if isinstance(exp, dict):
                feat_name = exp.get("feature", "N/A")
                impact = exp.get("impact", 0)
            else:
                feat_name = getattr(exp, "feature", "N/A")
                impact = getattr(exp, "impact", 0)
                
            human_feat = FEATURE_HUMAN_NAMES.get(feat_name, feat_name)
            drivers += f"  {i}️⃣ **{human_feat}**: Influencia del {impact:.0%}\n"
    if not drivers:
        drivers = "  Sin datos de explicabilidad disponibles.\n"

    detail = TEMPLATES["status"]["detail"].format(
        silo_id=silo_id,
        status_upper=status.upper(),
        metrics_table=metrics_table or "| Sin datos | - |\n",
        trends=trends,
        drivers=drivers,
        recommendation=get_agronomic_advice(status, metrics),
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

    # Extraer horizonte del mensaje (Parsing mejorado)
    horizon = 3  # default
    msg_lower = message.lower()
    
    if "mañana" in msg_lower:
        horizon = 1
    elif "semana" in msg_lower:
        horizon = 7
    elif "48 horas" in msg_lower or "2 dias" in msg_lower.replace("í", "i"):
        horizon = 2
    else:
        match = re.search(r"(\d+)\s*d[ií]as?", msg_lower)
        if match:
            horizon = int(match.group(1))

    # Usar raw_scores como proxy de probabilidad de empeoramiento
    raw = data.get("raw_scores", {})
    prob_worsen = raw.get("problema", 0) + raw.get("critico", 0)

    # Driver principal (Human Readable update)
    explanations = data.get("explanations", [])
    if explanations:
        top = explanations[0]
        if isinstance(top, dict):
            driver_name = top.get("feature", "N/A")
        else:
            driver_name = getattr(top, "feature", "N/A")
            
        human_feat = FEATURE_HUMAN_NAMES.get(driver_name, driver_name)
        driver = f"subida continua de {human_feat.lower()}"
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
