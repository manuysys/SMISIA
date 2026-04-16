"""
AGRILION — Motor de Riesgo Agronómico
========================================

Sistema de scoring compuesto (0–100) que integra:
1. Reglas agronómicas (humedad + temperatura → hongos, CO2 → fermentación)
2. Anomalías estadísticas (proporción en ventana reciente)
3. Desviaciones del modelo LSTM (error de predicción)

Clasificación:
    0–30  → NORMAL
    30–70 → WARNING
    70–100 → CRITICAL
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

from .config import AGRO_THRESHOLDS, RISK_CONFIG, SENSOR_COLUMNS

logger = logging.getLogger(__name__)


class RiskEngine:
    """
    Motor de evaluación de riesgo para silobolsas.

    Calcula un score compuesto basado en múltiples factores
    ponderados, combinando reglas agronómicas con detección
    estadística y ML.

    Attributes
    ----------
    weights : dict
        Pesos asignados a cada factor de riesgo.
    levels : dict
        Definición de niveles de riesgo.
    """

    def __init__(self, weights: Dict = None, custom_thresholds: Dict = None):
        """
        Parameters
        ----------
        weights : dict, optional
            Pesos personalizados para cada factor.
        custom_thresholds : dict, optional
            Umbrales agronómicos personalizados.
        """
        self.weights = weights or RISK_CONFIG["weights"]
        self.levels = RISK_CONFIG["levels"]
        self.thresholds = custom_thresholds or AGRO_THRESHOLDS
        self.anomaly_window = RISK_CONFIG["anomaly_window"]

    def _score_humidity_temp(
        self,
        temperature: float,
        humidity: float,
    ) -> float:
        """
        Score de riesgo por combinación humedad-temperatura.

        La combinación de alta humedad y alta temperatura favorece
        el crecimiento de hongos y bacterias en el grano.

        Returns
        -------
        float
            Score parcial (0–100).
        """
        temp_thresh = self.thresholds["temperature"]
        hum_thresh = self.thresholds["humidity"]

        # Score de temperatura (escala 0–100)
        if temperature <= temp_thresh["optimal_max"]:
            temp_score = 0.0
        elif temperature <= temp_thresh["warning"]:
            # Gradiente lineal entre optimal y warning
            temp_score = (
                (temperature - temp_thresh["optimal_max"])
                / (temp_thresh["warning"] - temp_thresh["optimal_max"])
                * 50
            )
        elif temperature <= temp_thresh["critical"]:
            temp_score = 50 + (
                (temperature - temp_thresh["warning"])
                / (temp_thresh["critical"] - temp_thresh["warning"])
                * 50
            )
        else:
            temp_score = 100.0

        # Score de humedad (escala 0–100)
        if humidity <= hum_thresh["optimal_max"]:
            hum_score = 0.0
        elif humidity <= hum_thresh["warning"]:
            hum_score = (
                (humidity - hum_thresh["optimal_max"])
                / (hum_thresh["warning"] - hum_thresh["optimal_max"])
                * 50
            )
        elif humidity <= hum_thresh["critical"]:
            hum_score = 50 + (
                (humidity - hum_thresh["warning"])
                / (hum_thresh["critical"] - hum_thresh["warning"])
                * 50
            )
        else:
            hum_score = 100.0

        # Efecto combinado: la combinación es peor que la suma de partes
        # (efecto sinérgico)
        individual_avg = (temp_score + hum_score) / 2
        synergy_factor = 1.0 + 0.3 * (min(temp_score, hum_score) / 100.0)
        combined = min(individual_avg * synergy_factor, 100.0)

        return round(combined, 2)

    def _score_co2(self, co2: float) -> float:
        """
        Score de riesgo por nivel de CO2.

        CO2 elevado indica actividad biológica: respiración de hongos,
        fermentación, o descomposición del grano.

        Returns
        -------
        float
            Score parcial (0–100).
        """
        co2_thresh = self.thresholds["co2"]

        if co2 <= co2_thresh["baseline"]:
            return 0.0
        elif co2 <= co2_thresh["warning"]:
            return (
                (co2 - co2_thresh["baseline"])
                / (co2_thresh["warning"] - co2_thresh["baseline"])
                * 30
            )
        elif co2 <= co2_thresh["critical"]:
            return 30 + (
                (co2 - co2_thresh["warning"])
                / (co2_thresh["critical"] - co2_thresh["warning"])
                * 40
            )
        elif co2 <= co2_thresh["severe"]:
            return 70 + (
                (co2 - co2_thresh["critical"])
                / (co2_thresh["severe"] - co2_thresh["critical"])
                * 30
            )
        else:
            return 100.0

    def _score_anomalies(
        self,
        anomaly_flags: pd.Series,
        window: int = None,
    ) -> float:
        """
        Score basado en proporción de anomalías recientes.

        Returns
        -------
        float
            Score parcial (0–100).
        """
        window = window or self.anomaly_window

        # Usar las últimas N mediciones
        recent = anomaly_flags.tail(window)

        if len(recent) == 0:
            return 0.0

        anomaly_rate = recent.mean()  # Proporción de anomalías

        # Escalar: 0% anomalías → 0 score, 20%+ → 100 score
        score = min(anomaly_rate / 0.2 * 100, 100.0)

        return round(score, 2)

    def _score_lstm_deviation(
        self,
        prediction_errors: np.ndarray,
        baseline_mae: float = None,
    ) -> float:
        """
        Score basado en desviación del modelo LSTM.

        Compara el error de predicción reciente contra el error
        base (MAE del entrenamiento).

        Returns
        -------
        float
            Score parcial (0–100).
        """
        if prediction_errors is None or len(prediction_errors) == 0:
            return 0.0

        # Usar la media de errores recientes
        recent_error = np.mean(np.abs(prediction_errors[-self.anomaly_window:]))

        if baseline_mae is None or baseline_mae == 0:
            # Sin baseline, usar el propio error como referencia
            baseline_mae = np.mean(np.abs(prediction_errors))

        if baseline_mae == 0:
            return 0.0

        # Ratio: 1x = normal, 2x = warning, 4x+ = critical
        ratio = recent_error / baseline_mae

        if ratio <= 1.0:
            score = 0.0
        elif ratio <= 2.0:
            score = (ratio - 1.0) * 40  # 1x→0, 2x→40
        elif ratio <= 4.0:
            score = 40 + (ratio - 2.0) * 30  # 2x→40, 4x→100
        else:
            score = 100.0

        return round(min(score, 100.0), 2)

    def calculate_risk_score(
        self,
        sensor_data: Dict[str, float],
        anomaly_flags: pd.Series = None,
        prediction_errors: np.ndarray = None,
        baseline_mae: float = None,
    ) -> int:
        """
        Calcula el score de riesgo compuesto (0–100).

        Pondera cuatro factores:
        1. Combinación humedad + temperatura (30%)
        2. Nivel de CO2 (25%)
        3. Anomalías estadísticas (20%)
        4. Desviación LSTM (25%)

        Parameters
        ----------
        sensor_data : dict
            Valores actuales de sensores: {temperature, humidity, co2}.
        anomaly_flags : pd.Series, optional
            Serie booleana de flags de anomalía recientes.
        prediction_errors : np.ndarray, optional
            Array de errores de predicción recientes.
        baseline_mae : float, optional
            MAE base del modelo para referencia.

        Returns
        -------
        int
            Score de riesgo (0–100).
        """
        # Factor 1: Humedad + Temperatura
        temp = sensor_data.get("temperature", 25.0)
        hum = sensor_data.get("humidity", 60.0)
        score_ht = self._score_humidity_temp(temp, hum)

        # Factor 2: CO2
        co2 = sensor_data.get("co2", 400.0)
        score_co2 = self._score_co2(co2)

        # Factor 3: Anomalías estadísticas
        if anomaly_flags is not None:
            score_anom = self._score_anomalies(anomaly_flags)
        else:
            score_anom = 0.0

        # Factor 4: Desviación LSTM
        score_lstm = self._score_lstm_deviation(prediction_errors, baseline_mae)

        # Ponderación
        w = self.weights
        total_score = (
            w["humidity_temp_combo"] * score_ht
            + w["co2_level"] * score_co2
            + w["statistical_anomalies"] * score_anom
            + w["lstm_deviation"] * score_lstm
        )

        return int(round(min(max(total_score, 0), 100)))

    def classify_risk(self, score: int) -> str:
        """
        Clasifica el score en un nivel de riesgo.

        Parameters
        ----------
        score : int
            Score de riesgo (0–100).

        Returns
        -------
        str
            Nivel: 'NORMAL', 'WARNING', o 'CRITICAL'.
        """
        for level, bounds in self.levels.items():
            if bounds["min"] <= score < bounds["max"] or (
                level == "CRITICAL" and score >= bounds["min"]
            ):
                return level
        return "NORMAL"

    def get_risk_factors(
        self,
        sensor_data: Dict[str, float],
        anomaly_flags: pd.Series = None,
        prediction_errors: np.ndarray = None,
        baseline_mae: float = None,
    ) -> Dict:
        """
        Calcula el desglose detallado de factores de riesgo.

        Returns
        -------
        dict
            Diccionario con score total, nivel, y contribución de cada factor.
        """
        temp = sensor_data.get("temperature", 25.0)
        hum = sensor_data.get("humidity", 60.0)
        co2 = sensor_data.get("co2", 400.0)

        # Calcular cada factor
        score_ht = self._score_humidity_temp(temp, hum)
        score_co2 = self._score_co2(co2)

        score_anom = 0.0
        if anomaly_flags is not None:
            score_anom = self._score_anomalies(anomaly_flags)

        score_lstm = self._score_lstm_deviation(prediction_errors, baseline_mae)

        # Totales ponderados
        w = self.weights
        total = int(round(min(max(
            w["humidity_temp_combo"] * score_ht
            + w["co2_level"] * score_co2
            + w["statistical_anomalies"] * score_anom
            + w["lstm_deviation"] * score_lstm,
            0
        ), 100)))

        level = self.classify_risk(total)
        level_info = self.levels[level]

        return {
            "total_score": total,
            "level": level,
            "emoji": level_info["emoji"],
            "color": level_info["color"],
            "factors": {
                "humidity_temp": {
                    "raw_score": score_ht,
                    "weighted_score": round(w["humidity_temp_combo"] * score_ht, 2),
                    "weight": w["humidity_temp_combo"],
                    "detail": f"Temp={temp:.1f}°C, Hum={hum:.1f}%",
                },
                "co2": {
                    "raw_score": score_co2,
                    "weighted_score": round(w["co2_level"] * score_co2, 2),
                    "weight": w["co2_level"],
                    "detail": f"CO2={co2:.0f}ppm",
                },
                "statistical_anomalies": {
                    "raw_score": score_anom,
                    "weighted_score": round(w["statistical_anomalies"] * score_anom, 2),
                    "weight": w["statistical_anomalies"],
                    "detail": "Proporción de anomalías en ventana reciente",
                },
                "lstm_deviation": {
                    "raw_score": score_lstm,
                    "weighted_score": round(w["lstm_deviation"] * score_lstm, 2),
                    "weight": w["lstm_deviation"],
                    "detail": "Desviación del modelo de predicción",
                },
            },
            "sensor_values": sensor_data,
        }

    def evaluate_timeline(
        self,
        df: pd.DataFrame,
        anomaly_df: pd.DataFrame = None,
        prediction_errors: np.ndarray = None,
        baseline_mae: float = None,
        window_size: int = None,
    ) -> pd.DataFrame:
        """
        Evalúa el riesgo a lo largo del tiempo completo.

        Genera un score de riesgo para cada timestamp de la serie.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos de sensores (DatetimeIndex).
        anomaly_df : pd.DataFrame, optional
            DataFrame con columna 'is_anomaly'.
        prediction_errors : np.ndarray, optional
            Errores de predicción.
        baseline_mae : float, optional
            MAE base.
        window_size : int, optional
            Ventana para cálculo de anomalías rolling.

        Returns
        -------
        pd.DataFrame
            DataFrame con columnas risk_score y risk_level.
        """
        window_size = window_size or self.anomaly_window
        result = df.copy()

        scores = []
        levels = []

        for i in range(len(df)):
            # Datos de sensores actuales
            sensor_data = {}
            for col in SENSOR_COLUMNS:
                if col in df.columns:
                    sensor_data[col] = df[col].iloc[i]

            # Anomalías recientes
            anomaly_flags = None
            if anomaly_df is not None and "is_anomaly" in anomaly_df.columns:
                start_idx = max(0, i - window_size)
                anomaly_flags = anomaly_df["is_anomaly"].iloc[start_idx : i + 1]

            # Errores de predicción recientes
            pred_err = None
            if prediction_errors is not None:
                # Alinear con los datos (prediction_errors puede ser más corto)
                offset = len(df) - len(prediction_errors)
                err_idx = i - offset
                if 0 <= err_idx < len(prediction_errors):
                    start_err = max(0, err_idx - window_size)
                    pred_err = prediction_errors[start_err : err_idx + 1]

            score = self.calculate_risk_score(
                sensor_data, anomaly_flags, pred_err, baseline_mae
            )
            level = self.classify_risk(score)

            scores.append(score)
            levels.append(level)

        result["risk_score"] = scores
        result["risk_level"] = levels

        # Estadísticas
        level_counts = pd.Series(levels).value_counts()
        logger.info("📊 Distribución de Riesgo:")
        for level in ["NORMAL", "WARNING", "CRITICAL"]:
            count = level_counts.get(level, 0)
            pct = count / len(levels) * 100
            emoji = self.levels[level]["emoji"]
            logger.info(f"   {emoji} {level}: {count} ({pct:.1f}%)")

        return result
