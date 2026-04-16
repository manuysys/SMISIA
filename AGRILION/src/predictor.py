"""
AGRILION — Predictor
======================

Módulo de predicción que utiliza el modelo LSTM entrenado para:
- Predecir el próximo valor de sensores (single-step)
- Predecir múltiples pasos futuros (multi-step recursive)
- Evaluar predicciones contra valores reales
- Detectar desviaciones significativas entre predicción y realidad
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

from .config import SENSOR_COLUMNS, LSTM_CONFIG
from .lstm_model import AgrilionLSTM
from .preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


class Predictor:
    """
    Predictor de series temporales para sensores agrícolas.

    Utiliza un modelo LSTM entrenado y un scaler para generar
    predicciones en escala original y detectar desviaciones.

    Attributes
    ----------
    model : AgrilionLSTM
        Modelo LSTM entrenado.
    preprocessor : DataPreprocessor
        Preprocesador con scaler ajustado.
    feature_columns : list
        Columnas de features.
    """

    def __init__(
        self,
        model: AgrilionLSTM,
        preprocessor: DataPreprocessor,
        feature_columns: List[str] = None,
    ):
        """
        Parameters
        ----------
        model : AgrilionLSTM
            Modelo LSTM entrenado.
        preprocessor : DataPreprocessor
            Preprocesador con scaler ajustado.
        feature_columns : list of str
            Columnas de features del modelo.
        """
        self.model = model
        self.preprocessor = preprocessor
        self.feature_columns = feature_columns or SENSOR_COLUMNS

    def predict_next(
        self,
        sequence: np.ndarray,
        return_original_scale: bool = True,
    ) -> np.ndarray:
        """
        Predice el próximo valor dado una secuencia de entrada.

        Parameters
        ----------
        sequence : np.ndarray
            Secuencia normalizada, shape (seq_length, n_features) o
            (1, seq_length, n_features).
        return_original_scale : bool
            Si True, devuelve predicción en escala original.

        Returns
        -------
        np.ndarray
            Predicción, shape (n_features,).
        """
        # Asegurar dimensión de batch
        if sequence.ndim == 2:
            sequence = sequence.reshape(1, *sequence.shape)

        # Predecir
        prediction_normalized = self.model.predict(sequence)

        if return_original_scale:
            prediction = self.preprocessor.inverse_transform(prediction_normalized)
            return prediction.flatten()
        else:
            return prediction_normalized.flatten()

    def predict_multistep(
        self,
        initial_sequence: np.ndarray,
        steps: int = 6,
        return_original_scale: bool = True,
    ) -> np.ndarray:
        """
        Predicción recursiva de múltiples pasos futuros.

        Usa cada predicción como entrada para el siguiente paso
        (autoregressive forecasting).

        Parameters
        ----------
        initial_sequence : np.ndarray
            Secuencia inicial normalizada, shape (seq_length, n_features).
        steps : int
            Número de pasos futuros a predecir.
        return_original_scale : bool
            Si True, devuelve en escala original.

        Returns
        -------
        np.ndarray
            Predicciones futuras, shape (steps, n_features).
        """
        current_sequence = initial_sequence.copy()
        predictions = []

        for step in range(steps):
            # Predecir el siguiente valor
            pred = self.predict_next(current_sequence, return_original_scale=False)
            predictions.append(pred)

            # Actualizar secuencia: eliminar el primer timestep, agregar predicción
            current_sequence = np.vstack([current_sequence[1:], pred.reshape(1, -1)])

        predictions = np.array(predictions)

        if return_original_scale:
            predictions = self.preprocessor.inverse_transform(predictions)

        logger.info(f"Predicción multi-step: {steps} pasos generados")
        return predictions

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: List[str] = None,
    ) -> Dict:
        """
        Evalúa las predicciones contra valores reales.

        Calcula métricas por feature:
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
        - R² (Coefficient of Determination)
        - MAPE (Mean Absolute Percentage Error)

        Parameters
        ----------
        y_true : np.ndarray
            Valores reales, shape (n_samples, n_features).
        y_pred : np.ndarray
            Predicciones, shape (n_samples, n_features).
        feature_names : list of str
            Nombres de las features.

        Returns
        -------
        dict
            Métricas de evaluación por feature y globales.
        """
        feature_names = feature_names or self.feature_columns

        # Asegurar 2D
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        n_features = min(y_true.shape[1], y_pred.shape[1], len(feature_names))
        metrics = {"per_feature": {}, "global": {}}

        for i in range(n_features):
            name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            true_i = y_true[:, i]
            pred_i = y_pred[:, i]

            mae = mean_absolute_error(true_i, pred_i)
            rmse = np.sqrt(mean_squared_error(true_i, pred_i))
            r2 = r2_score(true_i, pred_i)

            # MAPE (evitando división por cero)
            mask = true_i != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((true_i[mask] - pred_i[mask]) / true_i[mask])) * 100
            else:
                mape = float("inf")

            metrics["per_feature"][name] = {
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4),
                "MAPE": round(mape, 2),
            }

        # Métricas globales (promedio)
        all_mae = [m["MAE"] for m in metrics["per_feature"].values()]
        all_r2 = [m["R2"] for m in metrics["per_feature"].values()]

        metrics["global"] = {
            "avg_MAE": round(np.mean(all_mae), 4),
            "avg_R2": round(np.mean(all_r2), 4),
        }

        logger.info("📊 Métricas de Predicción:")
        for name, m in metrics["per_feature"].items():
            logger.info(
                f"   {name:>12}: MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, "
                f"R²={m['R2']:.4f}, MAPE={m['MAPE']:.2f}%"
            )

        return metrics

    def detect_prediction_anomalies(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold_factor: float = 2.0,
        feature_names: List[str] = None,
    ) -> pd.DataFrame:
        """
        Detecta anomalías basadas en desviación de la predicción LSTM.

        Un punto es anómalo si |real - predicho| > threshold_factor × mae_promedio.

        Parameters
        ----------
        y_true : np.ndarray
            Valores reales.
        y_pred : np.ndarray
            Predicciones del modelo.
        threshold_factor : float
            Multiplicador del error medio para determinar anomalía.
        feature_names : list of str
            Nombres de features.

        Returns
        -------
        pd.DataFrame
            DataFrame con flags de anomalía por predicción.
        """
        feature_names = feature_names or self.feature_columns

        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        n_features = min(y_true.shape[1], y_pred.shape[1], len(feature_names))

        result = pd.DataFrame()

        for i in range(n_features):
            name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            true_i = y_true[:, i]
            pred_i = y_pred[:, i]

            # Error absoluto
            abs_error = np.abs(true_i - pred_i)
            mean_error = np.mean(abs_error)
            threshold = threshold_factor * mean_error

            result[f"{name}_actual"] = true_i
            result[f"{name}_predicted"] = np.round(pred_i, 4)
            result[f"{name}_error"] = np.round(abs_error, 4)
            result[f"{name}_anomaly_lstm"] = abs_error > threshold

        # Flag global
        anomaly_cols = [c for c in result.columns if c.endswith("_anomaly_lstm")]
        result["is_prediction_anomaly"] = result[anomaly_cols].any(axis=1)

        n_anomalies = result["is_prediction_anomaly"].sum()
        logger.info(
            f"Anomalías LSTM: {n_anomalies}/{len(result)} puntos "
            f"(threshold_factor={threshold_factor})"
        )

        return result

    def generate_forecast_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: pd.DatetimeIndex = None,
        feature_names: List[str] = None,
    ) -> Dict:
        """
        Genera un reporte completo de predicción.

        Incluye métricas, anomalías detectadas y resumen ejecutivo.

        Parameters
        ----------
        y_true : np.ndarray
            Valores reales.
        y_pred : np.ndarray
            Predicciones.
        timestamps : pd.DatetimeIndex
            Timestamps correspondientes.
        feature_names : list of str
            Nombres de features.

        Returns
        -------
        dict
            Reporte con métricas, anomalías y resumen.
        """
        feature_names = feature_names or self.feature_columns

        # Métricas
        metrics = self.evaluate_predictions(y_true, y_pred, feature_names)

        # Anomalías de predicción
        pred_anomalies = self.detect_prediction_anomalies(
            y_true, y_pred, feature_names=feature_names
        )

        if timestamps is not None:
            pred_anomalies.index = timestamps[:len(pred_anomalies)]

        report = {
            "metrics": metrics,
            "prediction_anomalies": pred_anomalies,
            "n_prediction_anomalies": pred_anomalies["is_prediction_anomaly"].sum(),
            "total_predictions": len(pred_anomalies),
            "anomaly_rate": round(
                pred_anomalies["is_prediction_anomaly"].mean() * 100, 2
            ),
        }

        logger.info(
            f"\n📋 REPORTE DE PREDICCIÓN:\n"
            f"   Total predicciones: {report['total_predictions']}\n"
            f"   Anomalías detectadas: {report['n_prediction_anomalies']}\n"
            f"   Tasa de anomalía: {report['anomaly_rate']}%"
        )

        return report
