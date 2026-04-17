"""
AGRILION — Modelo LSTM para Series Temporales
================================================

Modelo de red neuronal LSTM para predicción de valores de sensores.
Soporta entradas multivariables y tanto predicción single-step
como multi-step forecasting.
"""

import numpy as np
import os
import logging
from typing import List, Tuple, Optional, Dict
from pathlib import Path

# Suprimir logs verbosos de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.optimizers import Adam

from .config import LSTM_CONFIG, DEFAULT_MODEL_PATH

logger = logging.getLogger(__name__)


class AgrilionLSTM:
    """
    Modelo LSTM para predicción de series temporales de sensores agrícolas.

    Arquitectura configurable con múltiples capas LSTM, dropout,
    y capa densa final. Soporta predicción univariable y multivariable.

    Attributes
    ----------
    model : keras.Model
        Modelo Keras compilado.
    history : keras.callbacks.History
        Historial de entrenamiento.
    config : dict
        Configuración del modelo.
    """

    def __init__(
        self,
        sequence_length: int = None,
        n_features: int = 3,
        lstm_units: List[int] = None,
        dropout_rate: float = None,
        dense_units: int = None,
        learning_rate: float = None,
    ):
        """
        Parameters
        ----------
        sequence_length : int
            Longitud de las secuencias de entrada.
        n_features : int
            Número de features de entrada (sensores).
        lstm_units : list of int
            Neuronas por capa LSTM.
        dropout_rate : float
            Tasa de dropout.
        dense_units : int
            Neuronas de la capa densa intermedia.
        learning_rate : float
            Tasa de aprendizaje.
        """
        cfg = LSTM_CONFIG
        self.sequence_length = sequence_length or cfg["sequence_length"]
        self.n_features = n_features
        self.lstm_units = lstm_units or cfg["lstm_units"]
        self.dropout_rate = dropout_rate or cfg["dropout_rate"]
        self.dense_units = dense_units or cfg["dense_units"]
        self.learning_rate = learning_rate or cfg["learning_rate"]

        self.model = None
        self.history = None
        self.config = {
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "dense_units": self.dense_units,
            "learning_rate": self.learning_rate,
        }

    def build_model(self, output_dim: int = None) -> keras.Model:
        """
        Construye la arquitectura del modelo LSTM.

        Arquitectura:
            Input → LSTM(128, return_sequences) → Dropout(0.2)
                  → LSTM(64) → Dropout(0.2)
                  → Dense(32, relu) → Dense(output_dim)

        Parameters
        ----------
        output_dim : int, optional
            Dimensión de salida. Default: n_features.

        Returns
        -------
        keras.Model
            Modelo construido (no compilado).
        """
        output_dim = output_dim or self.n_features

        model = Sequential(name="AGRILION_LSTM")

        # Input layer
        model.add(Input(shape=(self.sequence_length, self.n_features)))

        # Primera capa LSTM (return_sequences para apilar LSTMs)
        model.add(
            LSTM(
                units=self.lstm_units[0],
                return_sequences=len(self.lstm_units) > 1,
                name="lstm_1",
            )
        )
        model.add(Dropout(self.dropout_rate, name="dropout_1"))

        # Capas LSTM adicionales
        for i, units in enumerate(self.lstm_units[1:], start=2):
            return_seq = i < len(self.lstm_units)  # última capa no retorna secuencias
            model.add(
                LSTM(
                    units=units,
                    return_sequences=return_seq,
                    name=f"lstm_{i}",
                )
            )
            model.add(Dropout(self.dropout_rate, name=f"dropout_{i}"))

        # Capa densa intermedia
        model.add(Dense(self.dense_units, activation="relu", name="dense_hidden"))

        # Capa de salida
        model.add(Dense(output_dim, name="output"))

        self.model = model

        logger.info("=" * 60)
        logger.info("MODELO LSTM CONSTRUIDO")
        logger.info("=" * 60)
        model.summary(print_fn=logger.info)

        return model

    def build_multistep_model(
        self,
        forecast_horizon: int = 6,
        output_features: int = None,
    ) -> keras.Model:
        """
        Construye un modelo para predicción multi-step.

        La salida tiene shape (forecast_horizon, output_features).

        Parameters
        ----------
        forecast_horizon : int
            Número de pasos futuros a predecir.
        output_features : int
            Número de features de salida. Default: n_features.

        Returns
        -------
        keras.Model
            Modelo multi-step construido.
        """
        output_features = output_features or self.n_features
        output_dim = forecast_horizon * output_features

        model = self.build_model(output_dim=output_dim)

        # Reshape output para (batch, horizon, features)
        # (se maneja externamente en el predictor)

        logger.info(
            f"Modelo multi-step: horizon={forecast_horizon}, "
            f"output_features={output_features}"
        )

        return model

    def compile_model(self, model: keras.Model = None):
        """
        Compila el modelo con optimizer Adam y loss MSE.

        Parameters
        ----------
        model : keras.Model, optional
            Modelo a compilar. Default: self.model.
        """
        model = model or self.model
        if model is None:
            raise RuntimeError("Modelo no construido. Llame a build_model() primero.")

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        logger.info(
            f"Modelo compilado: optimizer=Adam(lr={self.learning_rate}), "
            f"loss=MSE, metrics=[MAE]"
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = None,
        batch_size: int = None,
        validation_split: float = None,
        verbose: int = 1,
    ) -> Dict:
        """
        Entrena el modelo LSTM.

        Incluye callbacks de Early Stopping, Reduce LR on Plateau
        y Model Checkpoint para entrenamiento robusto.

        Parameters
        ----------
        X_train : np.ndarray
            Secuencias de entrada, shape (n_samples, seq_length, n_features).
        y_train : np.ndarray
            Valores target, shape (n_samples, output_dim).
        epochs : int
            Número máximo de épocas.
        batch_size : int
            Tamaño de batch.
        validation_split : float
            Porcentaje de datos para validación interna.
        verbose : int
            Nivel de verbosidad (0=silencioso, 1=barra, 2=una línea por época).

        Returns
        -------
        dict
            Historial de entrenamiento con métricas.
        """
        if self.model is None:
            raise RuntimeError("Modelo no construido/compilado.")

        cfg = LSTM_CONFIG
        epochs = epochs or cfg["epochs"]
        batch_size = batch_size or cfg["batch_size"]
        validation_split = validation_split or cfg["validation_split"]

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=cfg["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=cfg["reduce_lr_factor"],
                patience=cfg["reduce_lr_patience"],
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        logger.info("=" * 60)
        logger.info("ENTRENAMIENTO LSTM")
        logger.info(f"  Épocas: {epochs}, Batch: {batch_size}")
        logger.info(f"  X_train shape: {X_train.shape}")
        logger.info(f"  y_train shape: {y_train.shape}")
        logger.info("=" * 60)

        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=False,  # No mezclar para series temporales
        )

        # Métricas finales
        final_loss = self.history.history["loss"][-1]
        final_val_loss = self.history.history["val_loss"][-1]
        best_epoch = np.argmin(self.history.history["val_loss"]) + 1
        total_epochs = len(self.history.history["loss"])

        logger.info(f"\n✅ Entrenamiento completado:")
        logger.info(f"   Épocas ejecutadas: {total_epochs}/{epochs}")
        logger.info(f"   Mejor época: {best_epoch}")
        logger.info(f"   Loss final: {final_loss:.6f}")
        logger.info(f"   Val Loss final: {final_val_loss:.6f}")

        return {
            "final_loss": final_loss,
            "final_val_loss": final_val_loss,
            "best_epoch": best_epoch,
            "total_epochs": total_epochs,
            "history": self.history.history,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Genera predicciones con el modelo entrenado.

        Parameters
        ----------
        X : np.ndarray
            Secuencias de entrada, shape (n_samples, seq_length, n_features).

        Returns
        -------
        np.ndarray
            Predicciones, shape (n_samples, output_dim).
        """
        if self.model is None:
            raise RuntimeError("Modelo no disponible.")

        predictions = self.model.predict(X, verbose=0)
        return predictions

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """
        Evalúa el modelo en datos de test.

        Parameters
        ----------
        X_test : np.ndarray
            Secuencias de test.
        y_test : np.ndarray
            Valores reales de test.

        Returns
        -------
        dict
            Métricas de evaluación: loss, mae.
        """
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = {
            "test_loss": results[0],
            "test_mae": results[1],
        }

        logger.info(f"📊 Evaluación: Loss={metrics['test_loss']:.6f}, MAE={metrics['test_mae']:.6f}")
        return metrics

    def save(self, path: str = None):
        """
        Guarda el modelo entrenado.

        Parameters
        ----------
        path : str, optional
            Ruta para guardar. Default: DEFAULT_MODEL_PATH.
        """
        path = Path(path) if path else DEFAULT_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(path)
        logger.info(f"💾 Modelo guardado en: {path}")

    def load(self, path: str = None):
        """
        Carga un modelo previamente entrenado.

        Parameters
        ----------
        path : str, optional
            Ruta del modelo. Default: DEFAULT_MODEL_PATH.
        """
        path = Path(path) if path else DEFAULT_MODEL_PATH

        if not path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {path}")

        self.model = keras_load_model(path)
        logger.info(f"📂 Modelo cargado desde: {path}")

    def get_model_summary(self) -> str:
        """Retorna un resumen del modelo como string."""
        if self.model is None:
            return "Modelo no construido."

        lines = []
        self.model.summary(print_fn=lambda x: lines.append(x))
        return "\n".join(lines)
