"""
AGRILION — Detección de Anomalías
====================================

Tres métodos complementarios para detección robusta:
1. Z-Score: detección estadística paramétrica
2. Moving Average: desviación de tendencia local
3. Isolation Forest: detección ML no supervisada

Los resultados se combinan mediante consenso para reducir falsos positivos.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.ensemble import IsolationForest
import logging

from .config import ANOMALY_CONFIG, SENSOR_COLUMNS

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detector de anomalías multimethod para series temporales de sensores.

    Combina tres métodos independientes y produce un consenso final
    que requiere que al menos N métodos coincidan para marcar una anomalía.

    Attributes
    ----------
    results : dict
        Resultados de cada método de detección.
    combined : pd.DataFrame
        Resultado combinado con consenso.
    """

    def __init__(
        self,
        zscore_threshold: float = None,
        ma_window: int = None,
        ma_threshold_sigma: float = None,
        if_contamination: float = None,
        consensus_min: int = None,
    ):
        """
        Parameters
        ----------
        zscore_threshold : float
            Umbral Z-score para marcar anomalía.
        ma_window : int
            Tamaño de ventana para media móvil.
        ma_threshold_sigma : float
            Número de sigmas para umbral de media móvil.
        if_contamination : float
            Proporción esperada de anomalías para Isolation Forest.
        consensus_min : int
            Mínimo de métodos que deben coincidir para consenso.
        """
        cfg = ANOMALY_CONFIG
        self.zscore_threshold = zscore_threshold or cfg["zscore_threshold"]
        self.ma_window = ma_window or cfg["moving_avg_window"]
        self.ma_threshold_sigma = ma_threshold_sigma or cfg["moving_avg_threshold_sigma"]
        self.if_contamination = if_contamination or cfg["isolation_forest_contamination"]
        self.consensus_min = consensus_min or cfg["consensus_min_methods"]

        self.results = {}
        self.combined = None
        self._isolation_forests = {}

    def detect_zscore(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
    ) -> pd.DataFrame:
        """
        Detección de anomalías por Z-Score.

        Calcula el Z-score para cada valor y marca como anomalía
        aquellos que superen el umbral configurado.

        Z-score = |x - μ| / σ

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos de sensores (DatetimeIndex).
        columns : list of str
            Columnas a analizar.

        Returns
        -------
        pd.DataFrame
            DataFrame con columnas adicionales:
            - {col}_zscore: Z-score calculado
            - {col}_anomaly_zscore: bool indicando anomalía
        """
        columns = columns or [c for c in SENSOR_COLUMNS if c in df.columns]
        result = df.copy()

        total_anomalies = 0
        for col in columns:
            values = df[col].values
            mean = np.nanmean(values)
            std = np.nanstd(values)

            if std == 0:
                logger.warning(f"Z-score: {col} tiene std=0, saltando.")
                result[f"{col}_zscore"] = 0.0
                result[f"{col}_anomaly_zscore"] = False
                continue

            z_scores = np.abs((values - mean) / std)
            is_anomaly = z_scores > self.zscore_threshold

            result[f"{col}_zscore"] = np.round(z_scores, 4)
            result[f"{col}_anomaly_zscore"] = is_anomaly

            n = is_anomaly.sum()
            total_anomalies += n
            logger.info(f"Z-score [{col}]: {n} anomalías (umbral={self.zscore_threshold})")

        self.results["zscore"] = result
        logger.info(f"Z-score total: {total_anomalies} anomalías detectadas")
        return result

    def detect_moving_average(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
    ) -> pd.DataFrame:
        """
        Detección por desviación de la media móvil.

        Calcula la media y desviación estándar con ventana deslizante,
        y marca como anomalía los valores fuera de μ ± (n_sigma × σ).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos de sensores.
        columns : list of str
            Columnas a analizar.

        Returns
        -------
        pd.DataFrame
            DataFrame con columnas adicionales de anomalía.
        """
        columns = columns or [c for c in SENSOR_COLUMNS if c in df.columns]
        result = df.copy()

        total_anomalies = 0
        for col in columns:
            # Media y std móviles
            rolling_mean = df[col].rolling(
                window=self.ma_window, center=True, min_periods=1
            ).mean()
            rolling_std = df[col].rolling(
                window=self.ma_window, center=True, min_periods=1
            ).std()

            # Reemplazar std=0 con un mínimo
            rolling_std = rolling_std.replace(0, rolling_std[rolling_std > 0].min())

            # Desviación respecto a la media móvil
            deviation = np.abs(df[col] - rolling_mean)
            threshold = self.ma_threshold_sigma * rolling_std

            is_anomaly = deviation > threshold

            result[f"{col}_ma_mean"] = np.round(rolling_mean, 4)
            result[f"{col}_ma_deviation"] = np.round(deviation, 4)
            result[f"{col}_anomaly_ma"] = is_anomaly

            n = is_anomaly.sum()
            total_anomalies += n
            logger.info(
                f"Moving Average [{col}]: {n} anomalías "
                f"(ventana={self.ma_window}, σ={self.ma_threshold_sigma})"
            )

        self.results["moving_average"] = result
        logger.info(f"Moving Average total: {total_anomalies} anomalías detectadas")
        return result

    def detect_isolation_forest(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
    ) -> pd.DataFrame:
        """
        Detección de anomalías con Isolation Forest.

        Algoritmo no supervisado basado en aislamiento aleatorio.
        Las anomalías son puntos que requieren menos particiones
        para ser aislados.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos de sensores.
        columns : list of str
            Columnas a analizar (se usa el conjunto multivariado).

        Returns
        -------
        pd.DataFrame
            DataFrame con scores y flags de anomalía.
        """
        columns = columns or [c for c in SENSOR_COLUMNS if c in df.columns]
        result = df.copy()

        # Preparar datos multivariados
        X = df[columns].values

        # Manejar NaN
        nan_mask = np.any(np.isnan(X), axis=1)
        X_clean = X[~nan_mask]

        if len(X_clean) < 10:
            logger.warning("Isolation Forest: datos insuficientes.")
            for col in columns:
                result[f"{col}_anomaly_if"] = False
            result["if_score"] = 0.0
            return result

        # Entrenar Isolation Forest
        iso_forest = IsolationForest(
            contamination=self.if_contamination,
            random_state=42,
            n_estimators=200,
            max_samples="auto",
            n_jobs=-1,
        )

        predictions = np.zeros(len(df))
        scores = np.zeros(len(df))

        predictions[~nan_mask] = iso_forest.fit_predict(X_clean)
        scores[~nan_mask] = iso_forest.decision_function(X_clean)

        # -1 = anomalía, 1 = normal
        is_anomaly = predictions == -1

        result["if_score"] = np.round(scores, 4)
        result["anomaly_if"] = is_anomaly

        # También marcar por columna individual
        for col in columns:
            # IF univariado por columna
            X_col = df[col].values.reshape(-1, 1)
            X_col_clean = X_col[~np.isnan(X_col).ravel()]

            if len(X_col_clean) > 10:
                iso_col = IsolationForest(
                    contamination=self.if_contamination,
                    random_state=42,
                    n_estimators=100,
                )
                pred_col = np.zeros(len(df))
                mask_col = ~np.isnan(X_col.ravel())
                pred_col[mask_col] = iso_col.fit_predict(X_col[mask_col])
                result[f"{col}_anomaly_if"] = pred_col == -1
            else:
                result[f"{col}_anomaly_if"] = False

        n_anomalies = is_anomaly.sum()
        self.results["isolation_forest"] = result
        self._isolation_forests["multivariate"] = iso_forest
        logger.info(
            f"Isolation Forest: {n_anomalies} anomalías "
            f"(contamination={self.if_contamination})"
        )

        return result

    def combine_anomalies(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
    ) -> pd.DataFrame:
        """
        Combina resultados de todos los métodos con consenso.

        Una anomalía se confirma si al menos `consensus_min` métodos
        la detectan simultáneamente.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame original.
        columns : list of str
            Columnas de sensores.

        Returns
        -------
        pd.DataFrame
            DataFrame con columnas de consenso por sensor y global.
        """
        columns = columns or [c for c in SENSOR_COLUMNS if c in df.columns]

        # Ejecutar los tres métodos si no se han ejecutado
        if "zscore" not in self.results:
            self.detect_zscore(df, columns)
        if "moving_average" not in self.results:
            self.detect_moving_average(df, columns)
        if "isolation_forest" not in self.results:
            self.detect_isolation_forest(df, columns)

        result = df.copy()
        methods = ["zscore", "ma", "if"]

        for col in columns:
            # Contar cuántos métodos detectan anomalía en cada punto
            method_flags = []
            for method in methods:
                col_name = f"{col}_anomaly_{method}"
                for method_result in self.results.values():
                    if col_name in method_result.columns:
                        method_flags.append(
                            method_result[col_name].astype(int).values
                        )
                        break

            if method_flags:
                votes = np.sum(method_flags, axis=0)
                result[f"{col}_anomaly_votes"] = votes
                result[f"{col}_anomaly_consensus"] = votes >= self.consensus_min
            else:
                result[f"{col}_anomaly_votes"] = 0
                result[f"{col}_anomaly_consensus"] = False

        # Flag global: cualquier sensor tiene anomalía por consenso
        consensus_cols = [f"{col}_anomaly_consensus" for col in columns]
        existing_consensus = [c for c in consensus_cols if c in result.columns]
        if existing_consensus:
            result["is_anomaly"] = result[existing_consensus].any(axis=1)
        else:
            result["is_anomaly"] = False

        n_consensus = result["is_anomaly"].sum()
        self.combined = result

        logger.info(
            f"Consenso final: {n_consensus} puntos anómalos "
            f"(≥{self.consensus_min} métodos)"
        )

        return result

    def detect_all(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
    ) -> pd.DataFrame:
        """
        Ejecuta todos los métodos de detección y devuelve el consenso.

        Función de conveniencia que llama a los tres métodos
        y produce el resultado combinado.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos de sensores.
        columns : list of str
            Columnas a analizar.

        Returns
        -------
        pd.DataFrame
            DataFrame con todas las columnas de anomalía y consenso.
        """
        columns = columns or [c for c in SENSOR_COLUMNS if c in df.columns]

        logger.info("=" * 60)
        logger.info("DETECCIÓN DE ANOMALÍAS — Ejecutando 3 métodos")
        logger.info("=" * 60)

        self.detect_zscore(df, columns)
        self.detect_moving_average(df, columns)
        self.detect_isolation_forest(df, columns)

        combined = self.combine_anomalies(df, columns)

        return combined

    def get_anomaly_summary(self) -> pd.DataFrame:
        """
        Genera un resumen de anomalías detectadas por timestamp.

        Returns
        -------
        pd.DataFrame
            Resumen con timestamps, sensores afectados y severidad.
        """
        if self.combined is None:
            raise RuntimeError("Ejecute detect_all() primero.")

        df = self.combined
        anomaly_rows = df[df["is_anomaly"]].copy()

        if len(anomaly_rows) == 0:
            logger.info("No se detectaron anomalías por consenso.")
            return pd.DataFrame()

        columns = [c for c in SENSOR_COLUMNS if c in df.columns]

        records = []
        for idx, row in anomaly_rows.iterrows():
            affected = []
            max_votes = 0
            for col in columns:
                votes_col = f"{col}_anomaly_votes"
                if votes_col in row and row[votes_col] >= self.consensus_min:
                    affected.append(col)
                    max_votes = max(max_votes, row[votes_col])

            records.append({
                "timestamp": idx,
                "affected_sensors": ", ".join(affected),
                "max_votes": max_votes,
                "severity": "HIGH" if max_votes == 3 else "MEDIUM",
                **{col: row[col] for col in columns if col in row},
            })

        summary = pd.DataFrame(records)
        logger.info(f"Resumen de anomalías: {len(summary)} puntos")
        return summary
