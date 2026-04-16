"""
AGRILION — Configuración Centralizada
======================================

Parámetros globales del sistema: umbrales de sensores, configuración LSTM,
niveles de riesgo y rutas de archivos.
"""

import os
from pathlib import Path

# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================

# Directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Directorios principales
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Archivos por defecto
DEFAULT_CSV_PATH = DATA_DIR / "sensor_data.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "lstm_agrilion.keras"
SCALER_PATH = MODELS_DIR / "scaler.joblib"

# =============================================================================
# PARÁMETROS DE SENSORES
# =============================================================================

# Columnas de sensores disponibles
SENSOR_COLUMNS = ["temperature", "humidity", "co2"]

# Rangos físicos válidos (para limpieza de datos)
SENSOR_RANGES = {
    "temperature": {"min": -10.0, "max": 60.0, "unit": "°C"},
    "humidity":    {"min": 0.0,   "max": 100.0, "unit": "%"},
    "co2":         {"min": 200.0, "max": 5000.0, "unit": "ppm"},
}

# Umbrales agronómicos de riesgo
AGRO_THRESHOLDS = {
    "temperature": {
        "optimal_min": 15.0,
        "optimal_max": 25.0,
        "warning": 30.0,      # Temperatura que favorece hongos
        "critical": 35.0,     # Degradación activa
    },
    "humidity": {
        "optimal_min": 40.0,
        "optimal_max": 60.0,
        "warning": 70.0,      # Riesgo de hongos
        "critical": 80.0,     # Condensación / deterioro severo
    },
    "co2": {
        "baseline": 400.0,    # CO2 ambiental normal
        "warning": 600.0,     # Actividad biológica detectada
        "critical": 800.0,    # Fermentación activa
        "severe": 1200.0,     # Situación crítica
    },
}

# =============================================================================
# PARÁMETROS DEL MODELO LSTM
# =============================================================================

LSTM_CONFIG = {
    # Arquitectura
    "sequence_length": 24,         # Ventana de entrada (24 mediciones = ~24 horas)
    "lstm_units": [128, 64],       # Neuronas por capa LSTM
    "dropout_rate": 0.2,           # Dropout para regularización
    "dense_units": 32,             # Neuronas de la capa densa intermedia

    # Entrenamiento
    "epochs": 50,                  # Épocas máximas
    "batch_size": 32,              # Tamaño de batch
    "learning_rate": 0.001,        # Tasa de aprendizaje inicial
    "validation_split": 0.1,       # Porcentaje de validación
    "early_stopping_patience": 10, # Paciencia para early stopping
    "reduce_lr_patience": 5,       # Paciencia para reducir LR
    "reduce_lr_factor": 0.5,       # Factor de reducción de LR

    # Datos
    "test_ratio": 0.2,            # Porcentaje de test
    "scaler_method": "minmax",     # 'minmax' o 'standard'
}

# =============================================================================
# DETECCIÓN DE ANOMALÍAS
# =============================================================================

ANOMALY_CONFIG = {
    "zscore_threshold": 3.0,                # Umbral Z-score
    "moving_avg_window": 12,                # Ventana para media móvil (12 horas)
    "moving_avg_threshold_sigma": 2.0,      # Sigmas para umbral de MA
    "isolation_forest_contamination": 0.05,  # Porcentaje esperado de anomalías
    "consensus_min_methods": 2,              # Mínimo de métodos para consenso
}

# =============================================================================
# MOTOR DE RIESGO
# =============================================================================

RISK_CONFIG = {
    # Pesos para scoring compuesto
    "weights": {
        "humidity_temp_combo": 0.30,    # Humedad alta + temperatura alta
        "co2_level": 0.25,             # Nivel de CO2
        "statistical_anomalies": 0.20,  # Anomalías estadísticas
        "lstm_deviation": 0.25,         # Desviación del modelo LSTM
    },

    # Clasificación de niveles
    "levels": {
        "NORMAL":   {"min": 0,  "max": 30,  "emoji": "🟢", "color": "#22c55e"},
        "WARNING":  {"min": 30, "max": 70,  "emoji": "🟡", "color": "#f59e0b"},
        "CRITICAL": {"min": 70, "max": 100, "emoji": "🔴", "color": "#ef4444"},
    },

    # Ventana de análisis para anomalías recientes (en mediciones)
    "anomaly_window": 24,
}

# =============================================================================
# DATOS SINTÉTICOS
# =============================================================================

SYNTHETIC_CONFIG = {
    "days": 30,                    # Días de datos a generar
    "interval_hours": 1,           # Intervalo entre mediciones (horas)
    "silo_id": "SILO_001",         # ID del silo
    "n_anomaly_events": 5,         # Número de eventos anómalos a inyectar
    "anomaly_duration_hours": 6,   # Duración de cada evento anómalo (horas)
    "random_seed": 42,             # Semilla para reproducibilidad

    # Parámetros de generación por sensor
    "temperature": {
        "base_mean": 25.0,
        "base_std": 2.0,
        "diurnal_amplitude": 5.0,      # Amplitud del ciclo diurno
        "anomaly_spike": 12.0,         # Incremento durante anomalía
    },
    "humidity": {
        "base_mean": 62.0,
        "base_std": 3.0,
        "diurnal_amplitude": 8.0,
        "anomaly_spike": 18.0,
    },
    "co2": {
        "base_mean": 420.0,
        "base_std": 20.0,
        "diurnal_amplitude": 30.0,
        "anomaly_spike": 400.0,
    },
}

# =============================================================================
# VISUALIZACIÓN
# =============================================================================

PLOT_CONFIG = {
    "figsize_timeseries": (16, 10),
    "figsize_predictions": (14, 6),
    "figsize_anomalies": (16, 10),
    "figsize_risk": (14, 5),
    "style": "dark_background",
    "dpi": 150,
    "save_format": "png",
    "color_palette": {
        "temperature": "#ff6b6b",
        "humidity": "#4ecdc4",
        "co2": "#ffd93d",
        "prediction": "#74b9ff",
        "anomaly": "#e74c3c",
        "normal": "#2ecc71",
        "warning": "#f39c12",
        "critical": "#e74c3c",
    },
}

# =============================================================================
# API
# =============================================================================

API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "title": "AGRILION API",
    "description": "API REST para análisis ML de silobolsas agrícolas",
    "version": "1.0.0",
}
