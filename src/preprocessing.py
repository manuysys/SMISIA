"""
AGRILION — Preprocesamiento de Datos
=======================================

Normalización, generación de secuencias para LSTM (windowing),
separación train/test temporal y transformación inversa.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import logging

from .config import LSTM_CONFIG, SENSOR_COLUMNS, SCALER_PATH

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesador de datos para el pipeline LSTM.

    Encapsula la normalización, generación de secuencias y
    separación de datos, manteniendo el estado del scaler para
    transformación inversa.

    Attributes
    ----------
    scaler : sklearn Scaler
        Scaler ajustado (MinMaxScaler o StandardScaler).
    method : str
        Método de normalización utilizado.
    feature_columns : list
        Columnas de features utilizadas.
    sequence_length : int
        Longitud de las secuencias para LSTM.
    """

    def __init__(
        self,
        method: str = None,
        sequence_length: int = None,
        feature_columns: List[str] = None,
    ):
        """
        Parameters
        ----------
        method : str
            Método de normalización: 'minmax' o 'standard'.
        sequence_length : int
            Longitud de las secuencias de entrada para LSTM.
        feature_columns : list of str
            Columnas a usar como features.
        """
        self.method = method or LSTM_CONFIG["scaler_method"]
        self.sequence_length = sequence_length or LSTM_CONFIG["sequence_length"]
        self.feature_columns = feature_columns or SENSOR_COLUMNS

        # Inicializar scaler
        if self.method == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif self.method == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Método de normalización no soportado: {self.method}")

        self._is_fitted = False

    def normalize(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> np.ndarray:
        """
        Normaliza los datos de sensores.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con columnas de sensores.
        fit : bool
            Si True, ajusta el scaler a los datos. Si False, usa scaler existente.

        Returns
        -------
        np.ndarray
            Array normalizado de shape (n_samples, n_features).
        """
        # Extraer columnas relevantes
        cols = [c for c in self.feature_columns if c in df.columns]
        data = df[cols].values.astype(np.float64)

        if fit:
            normalized = self.scaler.fit_transform(data)
            self._is_fitted = True
            logger.info(
                f"Scaler ajustado ({self.method}): "
                f"{len(cols)} features, {len(data)} samples"
            )
        else:
            if not self._is_fitted:
                raise RuntimeError("Scaler no ajustado. Llame a normalize(fit=True) primero.")
            normalized = self.scaler.transform(data)

        return normalized

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transforma datos normalizados de vuelta a la escala original.

        Parameters
        ----------
        data : np.ndarray
            Datos normalizados. Shape: (n_samples, n_features) o (n_features,).

        Returns
        -------
        np.ndarray
            Datos en escala original.
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler no ajustado.")

        # Manejar arrays 1D
        if data.ndim == 1:
            data = data.reshape(1, -1)

        return self.scaler.inverse_transform(data)

    def create_sequences(
        self,
        data: np.ndarray,
        target_indices: List[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera secuencias deslizantes para entrenamiento LSTM.

        Para cada posición i, crea:
        - X[i] = data[i : i + seq_length]           → secuencia de entrada
        - y[i] = data[i + seq_length, target_cols]   → valor siguiente a predecir

        Parameters
        ----------
        data : np.ndarray
            Datos normalizados de shape (n_samples, n_features).
        target_indices : list of int, optional
            Índices de las columnas target. Default: todas las columnas.

        Returns
        -------
        X : np.ndarray
            Secuencias de entrada, shape (n_sequences, seq_length, n_features).
        y : np.ndarray
            Valores target, shape (n_sequences, n_targets).
        """
        seq_len = self.sequence_length

        if len(data) <= seq_len:
            raise ValueError(
                f"Datos insuficientes: {len(data)} muestras para "
                f"secuencias de longitud {seq_len}"
            )

        if target_indices is None:
            target_indices = list(range(data.shape[1]))

        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i : i + seq_len])
            y.append(data[i + seq_len, target_indices])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        logger.info(
            f"Secuencias creadas: X={X.shape}, y={y.shape} "
            f"(seq_length={seq_len})"
        )

        return X, y

    def create_multistep_sequences(
        self,
        data: np.ndarray,
        forecast_horizon: int = 6,
        target_indices: List[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera secuencias para predicción multi-step.

        Parameters
        ----------
        data : np.ndarray
            Datos normalizados.
        forecast_horizon : int
            Número de pasos futuros a predecir.
        target_indices : list of int, optional
            Índices de columnas target.

        Returns
        -------
        X : np.ndarray
            Shape (n_sequences, seq_length, n_features).
        y : np.ndarray
            Shape (n_sequences, forecast_horizon, n_targets).
        """
        seq_len = self.sequence_length

        if target_indices is None:
            target_indices = list(range(data.shape[1]))

        X, y = [], []
        for i in range(len(data) - seq_len - forecast_horizon + 1):
            X.append(data[i : i + seq_len])
            y.append(data[i + seq_len : i + seq_len + forecast_horizon][:, target_indices])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        logger.info(
            f"Multi-step sequences: X={X.shape}, y={y.shape} "
            f"(horizon={forecast_horizon})"
        )

        return X, y

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_ratio: float = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Separación temporal train/test (sin shuffle para series temporales).

        Parameters
        ----------
        X : np.ndarray
            Secuencias de entrada.
        y : np.ndarray
            Valores target.
        test_ratio : float
            Proporción de datos para test. Default: config.

        Returns
        -------
        X_train, X_test, y_train, y_test : np.ndarray
            Datos separados temporalmente.
        """
        test_ratio = test_ratio or LSTM_CONFIG["test_ratio"]
        split_idx = int(len(X) * (1 - test_ratio))

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(
            f"Split temporal: train={len(X_train)}, test={len(X_test)} "
            f"(ratio={test_ratio})"
        )

        return X_train, X_test, y_train, y_test

    def save_scaler(self, path: Union[str, None] = None):
        """Guarda el scaler ajustado para uso futuro."""
        path = path or SCALER_PATH
        if self._is_fitted:
            joblib.dump(self.scaler, path)
            logger.info(f"Scaler guardado en: {path}")
        else:
            logger.warning("Scaler no ajustado, no se guardó.")

    def load_scaler(self, path: Union[str, None] = None):
        """Carga un scaler previamente guardado."""
        path = path or SCALER_PATH
        self.scaler = joblib.load(path)
        self._is_fitted = True
        logger.info(f"Scaler cargado desde: {path}")

    def prepare_pipeline(
        self,
        df: pd.DataFrame,
        test_ratio: float = None,
    ) -> dict:
        """
        Pipeline completo de preprocesamiento.

        Encadena: normalize → create_sequences → split_data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame limpio con datos de sensores.
        test_ratio : float
            Proporción de test.

        Returns
        -------
        dict
            Diccionario con:
            - X_train, X_test, y_train, y_test
            - normalized_data
            - n_features
        """
        # Normalizar
        normalized = self.normalize(df, fit=True)

        # Crear secuencias
        X, y = self.create_sequences(normalized)

        # Split
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_ratio)

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "normalized_data": normalized,
            "n_features": normalized.shape[1],
            "n_samples": len(normalized),
        }
