"""
AGRILION — Rule-Based AI Engine
=================================
Final fallback layer. NEVER fails. NEVER returns an error.
Generates structured Spanish responses using domain logic for grain storage.
No external dependencies required.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SiloState:
    temperature: float = 0.0
    humidity: float = 0.0
    co2: float = 0.0
    risk_score: int = 0
    risk_level: str = "NORMAL"
    silo_id: str = "SILO_001"
    alerts: list = None

    def __post_init__(self):
        if self.alerts is None:
            self.alerts = []


class RuleBasedEngine:
    """
    Deterministic rule engine for agricultural silo monitoring.

    Rules are evaluated in priority order. Multiple rules can fire
    simultaneously and their outputs are merged into a single structured response.
    """

    # ─── Thresholds ──────────────────────────────────────────────────────────
    TEMP_HIGH = 30.0
    TEMP_CRITICAL = 35.0
    HUM_HIGH = 70.0
    HUM_CRITICAL = 80.0
    CO2_HIGH = 700.0
    CO2_CRITICAL = 1200.0
    SCORE_WARNING = 40
    SCORE_HIGH = 60
    SCORE_CRITICAL = 80

    def respond(self, state: SiloState, user_message: str = "") -> str:
        """
        Generate a structured Spanish response based on silo state.
        Always succeeds — never raises exceptions.

        Args:
            state: current sensor readings and risk data
            user_message: original user question (used for routing)

        Returns:
            Formatted Spanish response string
        """
        try:
            return self._generate(state, user_message)
        except Exception:
            # Absolute last resort — should never reach this
            return (
                f"Estado: {state.risk_level}\n"
                f"Silo: {state.silo_id}\n\n"
                f"Temperatura: {state.temperature}°C | "
                f"Humedad: {state.humidity}% | "
                f"CO₂: {state.co2} ppm\n\n"
                "Monitoree el silo regularmente y consulte con un agrónomo."
            )

    def _generate(self, s: SiloState, message: str) -> str:
        problems, risks, actions, predictions = [], [], [], []

        # ─── Rule 1: Humidity + Temperature → Mold risk ──────────────────────
        if s.humidity >= self.HUM_CRITICAL and s.temperature >= self.TEMP_HIGH:
            problems.append("Humedad crítica combinada con temperatura elevada.")
            risks.append(
                "Condiciones óptimas para proliferación de hongos y micotoxinas. "
                "El grano puede deteriorarse irreversiblemente en 24–48 horas."
            )
            actions += [
                "Inspeccionar visualmente el silobolsa de inmediato.",
                "Verificar integridad del sellado y eliminar focos de humedad.",
                "Considerar extracción parcial del grano si el deterioro es visible.",
            ]
            predictions.append("Sin intervención, riesgo de pérdida total del lote en 2–5 días.")

        elif s.humidity >= self.HUM_HIGH and s.temperature >= self.TEMP_HIGH:
            problems.append("Humedad y temperatura simultáneamente elevadas.")
            risks.append(
                "Combinación favorable para el inicio de actividad fúngica. "
                "El equilibrio de humedad del grano puede verse comprometido."
            )
            actions += [
                "Aumentar frecuencia de monitoreo a cada 2 horas.",
                "Revisar puntos de condensación en la membrana del silobolsa.",
                "Preparar plan de acción si los valores continúan subiendo.",
            ]

        # ─── Rule 2: High humidity alone ──────────────────────────────────────
        elif s.humidity >= self.HUM_CRITICAL:
            problems.append(f"Humedad crítica: {s.humidity:.1f}%.")
            risks.append(
                "La humedad excesiva favorece la germinación del grano y el desarrollo "
                "de hongos incluso a temperaturas moderadas."
            )
            actions += [
                "Revisar sellado y posibles filtraciones en el silobolsa.",
                "Verificar que el terreno base no esté húmedo.",
                "Monitorear cada hora hasta estabilización.",
            ]

        elif s.humidity >= self.HUM_HIGH:
            problems.append(f"Humedad elevada: {s.humidity:.1f}%.")
            risks.append("Riesgo de inicio de actividad biológica y pérdida de calidad del grano.")
            actions += [
                "Revisar hermeticidad del silobolsa.",
                "Monitorear con mayor frecuencia.",
            ]

        # ─── Rule 3: CO2 → Biological / fermentation activity ─────────────────
        if s.co2 >= self.CO2_CRITICAL:
            problems.append(f"CO₂ en niveles críticos: {s.co2:.0f} ppm.")
            risks.append(
                "Nivel de CO₂ indica fermentación activa o respiración intensa de hongos. "
                "El grano está en proceso activo de deterioro."
            )
            actions += [
                "Extraer muestras del grano para evaluación urgente.",
                "Consultar a un ingeniero agrónomo antes de comercializar.",
                "Evaluar cierre total del acceso para evitar contaminación cruzada.",
            ]
            predictions.append("Fermentación activa puede reducir valor comercial del grano en un 40–70%.")

        elif s.co2 >= self.CO2_HIGH:
            problems.append(f"CO₂ elevado: {s.co2:.0f} ppm.")
            risks.append(
                "Actividad biológica por encima de lo normal. "
                "Posible inicio de fermentación o respiración fúngica."
            )
            actions += [
                "Aumentar frecuencia de lecturas de CO₂.",
                "Verificar temperatura interna del grano.",
            ]

        # ─── Rule 4: High temperature alone ──────────────────────────────────
        if s.temperature >= self.TEMP_CRITICAL and not problems:
            problems.append(f"Temperatura crítica: {s.temperature:.1f}°C.")
            risks.append(
                "A temperaturas superiores a 35°C el grano sufre daño térmico. "
                "Se acelera la degradación de proteínas y lípidos."
            )
            actions += [
                "Verificar exposición solar directa sobre el silobolsa.",
                "Considerar cubierta reflectante si la temperatura exterior es alta.",
                "Monitorear punto de condensación nocturno.",
            ]
        elif s.temperature >= self.TEMP_HIGH and not problems:
            problems.append(f"Temperatura elevada: {s.temperature:.1f}°C.")
            risks.append("Temperatura por encima del umbral recomendado para almacenamiento seguro.")
            actions += [
                "Monitorear evolución horaria.",
                "Revisar si hay fuente de calor externa.",
            ]

        # ─── Rule 5: Score-based escalation ───────────────────────────────────
        if s.risk_score >= self.SCORE_CRITICAL and not problems:
            problems.append("Múltiples factores de riesgo activos simultáneamente.")
            risks.append(
                "El sistema detecta una combinación de condiciones adversas "
                "que en conjunto representan alto peligro para el grano."
            )
            actions += [
                "Inspección presencial inmediata del silobolsa.",
                "Activar protocolo de emergencia de cosecha si aplica.",
            ]

        # ─── Rule 6: NORMAL state ─────────────────────────────────────────────
        if not problems:
            return self._format_normal(s)

        # ─── Deduplicate actions ──────────────────────────────────────────────
        seen = set()
        unique_actions = []
        for a in actions:
            if a not in seen:
                seen.add(a)
                unique_actions.append(a)

        return self._format_response(s, problems, risks, unique_actions, predictions)

    def _format_response(
        self,
        s: SiloState,
        problems: list,
        risks: list,
        actions: list,
        predictions: list,
    ) -> str:
        emoji = {"CRITICAL": "🔴", "WARNING": "⚠️"}.get(s.risk_level, "🟡")
        lines = [
            f"Estado: {s.risk_level} {emoji}",
            f"Silo: {s.silo_id}",
            f"Temperatura: {s.temperature:.1f}°C | Humedad: {s.humidity:.1f}% | CO₂: {s.co2:.0f} ppm",
            "",
            "Problema:",
            " ".join(problems),
            "",
            "Riesgo:",
            " ".join(risks),
        ]

        if actions:
            lines += ["", "Acciones recomendadas:"]
            lines += [f"• {a}" for a in actions[:4]]  # max 4 actions

        if predictions:
            lines += ["", "Predicción:", " ".join(predictions)]

        if s.risk_score >= self.SCORE_CRITICAL:
            lines += ["", "⚠️ ACCIÓN URGENTE requerida. No espere la próxima lectura programada."]

        return "\n".join(lines)

    def _format_normal(self, s: SiloState) -> str:
        return (
            f"Estado: NORMAL 🟢\n"
            f"Silo: {s.silo_id}\n"
            f"Temperatura: {s.temperature:.1f}°C | Humedad: {s.humidity:.1f}% | CO₂: {s.co2:.0f} ppm\n\n"
            "El silobolsa se encuentra dentro de los parámetros normales de almacenamiento.\n\n"
            "Acciones recomendadas:\n"
            "• Continuar con el monitoreo periódico programado.\n"
            "• Mantener el sellado del silobolsa en buenas condiciones.\n"
            "• Registrar las lecturas para detección de tendencias."
        )

    @classmethod
    def from_context(cls, context: dict) -> tuple["RuleBasedEngine", SiloState]:
        """Factory: create engine and state from a context dict."""
        sensors = context.get("current_sensors", {})
        state = SiloState(
            temperature=float(sensors.get("temperature", 0)),
            humidity=float(sensors.get("humidity", 0)),
            co2=float(sensors.get("co2", 0)),
            risk_score=int(context.get("risk_score") or 0),
            risk_level=context.get("risk_level", "NORMAL"),
            silo_id=context.get("silo_id", "SILO_001"),
            alerts=context.get("active_alerts", []),
        )
        return cls(), state
