"""
AGRILION — Sistema de Alertas Inteligentes
============================================

Genera alertas legibles en español basadas en:
- Score de riesgo calculado
- Factores de riesgo individuales
- Anomalías detectadas
- Predicciones del modelo LSTM

Las alertas incluyen tipo de riesgo, nivel, recomendación de acción
y formato para consola, logs y API.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging

from .config import AGRO_THRESHOLDS, RISK_CONFIG

logger = logging.getLogger(__name__)


class Alert:
    """Representa una alerta individual del sistema."""

    def __init__(
        self,
        level: str,
        category: str,
        message: str,
        detail: str = "",
        recommendation: str = "",
        timestamp: datetime = None,
        sensor_values: Dict = None,
        risk_score: int = 0,
    ):
        self.level = level  # NORMAL, WARNING, CRITICAL
        self.category = category  # hongos, fermentacion, anomalia_termica, etc.
        self.message = message
        self.detail = detail
        self.recommendation = recommendation
        self.timestamp = timestamp or datetime.now()
        self.sensor_values = sensor_values or {}
        self.risk_score = risk_score

    @property
    def emoji(self) -> str:
        """Emoji representativo del nivel."""
        return RISK_CONFIG["levels"].get(self.level, {}).get("emoji", "ℹ️")

    def to_dict(self) -> Dict:
        """Serializa la alerta como diccionario."""
        return {
            "level": self.level,
            "category": self.category,
            "message": self.message,
            "detail": self.detail,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            "sensor_values": self.sensor_values,
            "risk_score": self.risk_score,
        }

    def __repr__(self):
        return f"{self.emoji} {self.level} — {self.message}"


class AlertSystem:
    """
    Sistema de generación de alertas inteligentes para silobolsas.

    Analiza los factores de riesgo y genera alertas contextualizadas
    con recomendaciones de acción.

    Attributes
    ----------
    alerts_history : list
        Historial de todas las alertas generadas.
    """

    # Catálogo de alertas predefinidas
    ALERT_CATALOG = {
        "hongos": {
            "WARNING": {
                "message": "Riesgo de hongos detectado",
                "recommendation": (
                    "Verificar sellado del silobolsa. Considerar ventilación "
                    "si es posible. Monitorear evolución en las próximas 12 horas."
                ),
            },
            "CRITICAL": {
                "message": "Riesgo ALTO de proliferación de hongos",
                "recommendation": (
                    "ACCIÓN URGENTE: Inspeccionar el silobolsa inmediatamente. "
                    "Considerar apertura y descarga parcial. Evaluar tratamiento "
                    "con fungicida si el grano lo permite."
                ),
            },
        },
        "fermentacion": {
            "WARNING": {
                "message": "Indicios de actividad biológica detectados",
                "recommendation": (
                    "Monitorear niveles de CO2 frecuentemente. Si la tendencia "
                    "continúa al alza, planificar descarga preventiva."
                ),
            },
            "CRITICAL": {
                "message": "Fermentación activa detectada — CO2 muy elevado",
                "recommendation": (
                    "ACCIÓN URGENTE: La fermentación activa compromete la calidad "
                    "del grano. Proceder a descarga inmediata o ventilación forzada. "
                    "Contactar asesor agrónomo."
                ),
            },
        },
        "anomalia_termica": {
            "WARNING": {
                "message": "Anomalía térmica fuera del patrón normal",
                "recommendation": (
                    "Valor de temperatura fuera de lo esperado por el modelo. "
                    "Verificar que el sensor funcione correctamente. "
                    "Inspeccionar silobolsa por posibles daños."
                ),
            },
            "CRITICAL": {
                "message": "Anomalía térmica CRÍTICA — Calentamiento fuera de control",
                "recommendation": (
                    "Temperatura peligrosamente alta. Posible autodescalentamiento "
                    "del grano. Descarga urgente recomendada."
                ),
            },
        },
        "anomalia_humedad": {
            "WARNING": {
                "message": "Humedad anormalmente elevada detectada",
                "recommendation": (
                    "Verificar integridad del silobolsa (posible entrada de agua). "
                    "Monitorear condensación. Revisar puntos de sellado."
                ),
            },
            "CRITICAL": {
                "message": "Humedad CRÍTICA — Riesgo de deterioro severo",
                "recommendation": (
                    "Humedad extrema compromete la conservación del grano. "
                    "Inspección urgente. Verificar sellos y posible ingreso de agua."
                ),
            },
        },
        "prediccion_anomala": {
            "WARNING": {
                "message": "Valores desviados de la predicción del modelo",
                "recommendation": (
                    "Los valores reales difieren significativamente de los predichos. "
                    "Esto puede indicar un cambio de condición inesperado. "
                    "Aumentar frecuencia de monitoreo."
                ),
            },
            "CRITICAL": {
                "message": "Desviación EXTREMA del modelo predictivo",
                "recommendation": (
                    "Los sensores muestran comportamiento altamente anómalo "
                    "según el modelo ML. Inspección presencial recomendada "
                    "de forma urgente."
                ),
            },
        },
        "deterioro_general": {
            "WARNING": {
                "message": "Tendencia de deterioro gradual detectada",
                "recommendation": (
                    "Múltiples indicadores sugieren un deterioro gradual. "
                    "Planificar descarga dentro de los próximos 7 días "
                    "si las condiciones no mejoran."
                ),
            },
            "CRITICAL": {
                "message": "Deterioro generalizado — Múltiples sensores en alerta",
                "recommendation": (
                    "Situación crítica con múltiples factores de riesgo activos. "
                    "Descarga inmediata recomendada. Evaluar calidad del grano "
                    "antes de comercializar."
                ),
            },
        },
    }

    def __init__(self):
        self.alerts_history: List[Alert] = []

    def generate_alerts(
        self,
        risk_factors: Dict,
        anomaly_summary: pd.DataFrame = None,
        prediction_anomalies: pd.DataFrame = None,
        timestamp: datetime = None,
    ) -> List[Alert]:
        """
        Genera alertas basadas en factores de riesgo y anomalías.

        Parameters
        ----------
        risk_factors : dict
            Resultado de RiskEngine.get_risk_factors().
        anomaly_summary : pd.DataFrame, optional
            Resumen de anomalías detectadas.
        prediction_anomalies : pd.DataFrame, optional
            Anomalías de predicción LSTM.
        timestamp : datetime, optional
            Timestamp de la evaluación.

        Returns
        -------
        list of Alert
            Lista de alertas generadas.
        """
        alerts = []
        score = risk_factors.get("total_score", 0)
        level = risk_factors.get("level", "NORMAL")
        sensor_values = risk_factors.get("sensor_values", {})
        factors = risk_factors.get("factors", {})

        # Si es NORMAL y no hay anomalías, no generar alertas
        if level == "NORMAL" and (anomaly_summary is None or len(anomaly_summary) == 0):
            return alerts

        # --- Alerta por humedad + temperatura ---
        ht_factor = factors.get("humidity_temp", {})
        if ht_factor.get("raw_score", 0) > 30:
            temp = sensor_values.get("temperature", 0)
            hum = sensor_values.get("humidity", 0)
            alert_level = "CRITICAL" if ht_factor["raw_score"] > 70 else "WARNING"
            catalog = self.ALERT_CATALOG["hongos"][alert_level]

            alerts.append(Alert(
                level=alert_level,
                category="hongos",
                message=catalog["message"],
                detail=f"Temperatura: {temp:.1f}°C, Humedad: {hum:.1f}%",
                recommendation=catalog["recommendation"],
                timestamp=timestamp,
                sensor_values=sensor_values,
                risk_score=score,
            ))

        # --- Alerta por CO2 ---
        co2_factor = factors.get("co2", {})
        if co2_factor.get("raw_score", 0) > 30:
            co2 = sensor_values.get("co2", 0)
            alert_level = "CRITICAL" if co2_factor["raw_score"] > 70 else "WARNING"
            catalog = self.ALERT_CATALOG["fermentacion"][alert_level]

            alerts.append(Alert(
                level=alert_level,
                category="fermentacion",
                message=catalog["message"],
                detail=f"CO2: {co2:.0f} ppm",
                recommendation=catalog["recommendation"],
                timestamp=timestamp,
                sensor_values=sensor_values,
                risk_score=score,
            ))

        # --- Alerta por anomalías estadísticas ---
        anom_factor = factors.get("statistical_anomalies", {})
        if anom_factor.get("raw_score", 0) > 30:
            alert_level = "CRITICAL" if anom_factor["raw_score"] > 70 else "WARNING"

            # Determinar qué tipo de anomalía predomina
            if sensor_values.get("temperature", 25) > AGRO_THRESHOLDS["temperature"]["warning"]:
                category = "anomalia_termica"
            elif sensor_values.get("humidity", 60) > AGRO_THRESHOLDS["humidity"]["warning"]:
                category = "anomalia_humedad"
            else:
                category = "deterioro_general"

            catalog = self.ALERT_CATALOG[category][alert_level]

            alerts.append(Alert(
                level=alert_level,
                category=category,
                message=catalog["message"],
                detail=f"Score anomalía: {anom_factor['raw_score']:.0f}/100",
                recommendation=catalog["recommendation"],
                timestamp=timestamp,
                sensor_values=sensor_values,
                risk_score=score,
            ))

        # --- Alerta por desviación LSTM ---
        lstm_factor = factors.get("lstm_deviation", {})
        if lstm_factor.get("raw_score", 0) > 30:
            alert_level = "CRITICAL" if lstm_factor["raw_score"] > 70 else "WARNING"
            catalog = self.ALERT_CATALOG["prediccion_anomala"][alert_level]

            alerts.append(Alert(
                level=alert_level,
                category="prediccion_anomala",
                message=catalog["message"],
                detail=f"Desviación LSTM: {lstm_factor['raw_score']:.0f}/100",
                recommendation=catalog["recommendation"],
                timestamp=timestamp,
                sensor_values=sensor_values,
                risk_score=score,
            ))

        # --- Alerta general si score > 70 y no hay alertas específicas ---
        if score >= 70 and not alerts:
            catalog = self.ALERT_CATALOG["deterioro_general"]["CRITICAL"]
            alerts.append(Alert(
                level="CRITICAL",
                category="deterioro_general",
                message=catalog["message"],
                detail=f"Score de riesgo: {score}/100",
                recommendation=catalog["recommendation"],
                timestamp=timestamp,
                sensor_values=sensor_values,
                risk_score=score,
            ))

        # Guardar en historial
        self.alerts_history.extend(alerts)

        return alerts

    def format_alert_report(
        self,
        alerts: List[Alert],
        include_recommendations: bool = True,
    ) -> str:
        """
        Formatea alertas como texto legible para consola/log.

        Parameters
        ----------
        alerts : list of Alert
            Alertas a formatear.
        include_recommendations : bool
            Si True, incluye recomendaciones de acción.

        Returns
        -------
        str
            Texto formateado con alertas.
        """
        if not alerts:
            return "✅ Sin alertas activas — Todos los parámetros dentro de rangos normales."

        lines = []
        lines.append("=" * 70)
        lines.append("🚨  REPORTE DE ALERTAS — AGRILION")
        lines.append("=" * 70)

        for i, alert in enumerate(alerts, 1):
            lines.append("")
            lines.append(f"  {alert.emoji} ALERTA #{i}: {alert.level}")
            lines.append(f"  {'─' * 50}")
            lines.append(f"  📌 {alert.message}")
            lines.append(f"  📊 Categoría: {alert.category}")
            lines.append(f"  📈 Score de riesgo: {alert.risk_score}/100")

            if alert.detail:
                lines.append(f"  📋 Detalle: {alert.detail}")

            if alert.timestamp:
                ts = alert.timestamp
                if isinstance(ts, datetime):
                    lines.append(f"  🕐 Timestamp: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    lines.append(f"  🕐 Timestamp: {ts}")

            if include_recommendations and alert.recommendation:
                lines.append(f"  💡 Recomendación: {alert.recommendation}")

        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  Total alertas: {len(alerts)} "
                     f"(CRITICAL: {sum(1 for a in alerts if a.level == 'CRITICAL')}, "
                     f"WARNING: {sum(1 for a in alerts if a.level == 'WARNING')})")
        lines.append("=" * 70)

        return "\n".join(lines)

    def to_json(self, alerts: List[Alert]) -> str:
        """
        Serializa alertas a formato JSON para API.

        Parameters
        ----------
        alerts : list of Alert
            Alertas a serializar.

        Returns
        -------
        str
            JSON string con alertas.
        """
        return json.dumps(
            [alert.to_dict() for alert in alerts],
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    def get_alert_summary(self) -> Dict:
        """
        Resumen estadístico de todas las alertas generadas.

        Returns
        -------
        dict
            Conteos por nivel y categoría.
        """
        if not self.alerts_history:
            return {"total": 0, "by_level": {}, "by_category": {}}

        by_level = {}
        by_category = {}

        for alert in self.alerts_history:
            by_level[alert.level] = by_level.get(alert.level, 0) + 1
            by_category[alert.category] = by_category.get(alert.category, 0) + 1

        return {
            "total": len(self.alerts_history),
            "by_level": by_level,
            "by_category": by_category,
        }
