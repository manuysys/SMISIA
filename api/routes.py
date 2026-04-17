"""
AGRILION — API Routes
========================

Endpoints REST para integración del sistema ML de AGRILION:
- /predict → Predicción de valores futuros
- /analyze → Análisis completo (anomalías + riesgo + alertas)
- /health → Health check
- /model/status → Estado del modelo
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from pathlib import Path
import logging

from fastapi import APIRouter, HTTPException

from .schemas import (
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
    AnalysisRequest,
    AnalysisResponse,
    AnomalyResult,
    AlertResult,
    RiskResult,
    HealthResponse,
    ModelStatusResponse,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src import __version__
from src.config import DEFAULT_MODEL_PATH, SCALER_PATH, LSTM_CONFIG, SENSOR_COLUMNS
from src.lstm_model import AgrilionLSTM
from src.preprocessing import DataPreprocessor
from src.predictor import Predictor
from src.anomaly_detection import AnomalyDetector
from src.risk_engine import RiskEngine
from src.alerts import AlertSystem

logger = logging.getLogger(__name__)

router = APIRouter()

# ===========================================================================
# ESTADO GLOBAL (se inicializa al arrancar)
# ===========================================================================

_model: Optional[AgrilionLSTM] = None
_preprocessor: Optional[DataPreprocessor] = None
_predictor: Optional[Predictor] = None
_risk_engine = RiskEngine()
_alert_system = AlertSystem()
_anomaly_detector = AnomalyDetector()


def initialize_model():
    """Carga el modelo y scaler si existen."""
    global _model, _preprocessor, _predictor

    _preprocessor = DataPreprocessor()

    if Path(DEFAULT_MODEL_PATH).exists() and Path(SCALER_PATH).exists():
        try:
            _model = AgrilionLSTM(n_features=len(SENSOR_COLUMNS))
            _model.load(str(DEFAULT_MODEL_PATH))
            _preprocessor.load_scaler(str(SCALER_PATH))
            _predictor = Predictor(_model, _preprocessor)
            logger.info("✅ Modelo y scaler cargados para API")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar el modelo: {e}")
            _model = None
    else:
        logger.info("ℹ️ Modelo no encontrado. Entrene el modelo primero con main.py")


def _readings_to_df(readings) -> pd.DataFrame:
    """Convierte lista de SensorReading a DataFrame."""
    data = [
        {
            "timestamp": r.timestamp,
            "temperature": r.temperature,
            "humidity": r.humidity,
            "co2": r.co2,
            "silo_id": r.silo_id,
        }
        for r in readings
    ]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


# ===========================================================================
# ENDPOINTS
# ===========================================================================

@router.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health_check():
    """Verifica el estado del servicio."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        model_loaded=_model is not None and _model.model is not None,
        timestamp=datetime.now().isoformat(),
    )


@router.get("/model/status", response_model=ModelStatusResponse, tags=["Modelo"])
async def model_status():
    """Devuelve información sobre el modelo cargado."""
    if _model is None or _model.model is None:
        return ModelStatusResponse(
            model_loaded=False,
            model_path=str(DEFAULT_MODEL_PATH),
        )

    return ModelStatusResponse(
        model_loaded=True,
        model_path=str(DEFAULT_MODEL_PATH),
        architecture=_model.get_model_summary(),
        n_parameters=_model.model.count_params(),
    )


@router.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
async def predict(request: PredictionRequest):
    """
    Predice valores futuros de sensores.

    Requiere al menos 24 lecturas recientes (sequence_length).
    Devuelve predicciones para los próximos N pasos.
    """
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Entrene el modelo primero con main.py"
        )

    try:
        df = _readings_to_df(request.readings)
        sensor_cols = [c for c in SENSOR_COLUMNS if c in df.columns]

        # Normalizar
        normalized = _preprocessor.normalize(df, fit=False)

        # Tomar la última secuencia
        seq_len = LSTM_CONFIG["sequence_length"]
        if len(normalized) < seq_len:
            raise HTTPException(
                status_code=400,
                detail=f"Se necesitan al menos {seq_len} lecturas. Recibidas: {len(normalized)}"
            )

        last_sequence = normalized[-seq_len:]

        # Predecir multi-step
        predictions_raw = _predictor.predict_multistep(
            last_sequence, steps=request.steps, return_original_scale=True
        )

        # Formatear resultado
        last_ts = df.index[-1]
        freq = pd.infer_freq(df.index[-10:]) or "1h"

        prediction_results = []
        for i, pred in enumerate(predictions_raw):
            future_ts = last_ts + pd.Timedelta(freq) * (i + 1)
            prediction_results.append(PredictionResult(
                timestamp=future_ts.isoformat(),
                temperature=round(float(pred[0]), 2),
                humidity=round(float(pred[1]), 2) if len(pred) > 1 else 0.0,
                co2=round(float(pred[2]), 2) if len(pred) > 2 else 0.0,
            ))

        return PredictionResponse(
            status="success",
            predictions=prediction_results,
            model_info={
                "sequence_length": seq_len,
                "steps_predicted": request.steps,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.post("/analyze", response_model=AnalysisResponse, tags=["Análisis"])
async def analyze(request: AnalysisRequest):
    """
    Análisis completo: anomalías + riesgo + alertas.

    Combina detección de anomalías, evaluación de riesgo y
    generación de alertas en un solo endpoint.
    """
    try:
        df = _readings_to_df(request.readings)
        sensor_cols = [c for c in SENSOR_COLUMNS if c in df.columns]

        anomaly_results = []
        alert_results = []
        risk_result = None
        prediction_results = []

        # 1. Detección de anomalías
        anomaly_df = None
        if request.include_anomaly_detection and len(df) >= 10:
            anomaly_df = _anomaly_detector.detect_all(df, sensor_cols)

            for idx, row in anomaly_df.iterrows():
                if row.get("is_anomaly", False):
                    affected = [
                        col for col in sensor_cols
                        if row.get(f"{col}_anomaly_consensus", False)
                    ]
                    anomaly_results.append(AnomalyResult(
                        timestamp=str(idx),
                        is_anomaly=True,
                        affected_sensors=affected,
                        severity="HIGH" if row.get("temperature_anomaly_votes", 0) >= 3 else "MEDIUM",
                        methods_triggered=int(
                            max(row.get(f"{col}_anomaly_votes", 0) for col in sensor_cols)
                        ),
                    ))

        # 2. Evaluación de riesgo
        if request.include_risk_assessment:
            # Usar los últimos valores
            last_values = {col: float(df[col].iloc[-1]) for col in sensor_cols}

            anomaly_flags = None
            if anomaly_df is not None and "is_anomaly" in anomaly_df.columns:
                anomaly_flags = anomaly_df["is_anomaly"]

            risk_factors = _risk_engine.get_risk_factors(
                last_values, anomaly_flags=anomaly_flags
            )

            risk_result = RiskResult(
                total_score=risk_factors["total_score"],
                level=risk_factors["level"],
                emoji=risk_factors["emoji"],
                factors=risk_factors["factors"],
            )

            # 3. Alertas
            alerts = _alert_system.generate_alerts(risk_factors)
            alert_results = [
                AlertResult(
                    level=a.level,
                    category=a.category,
                    message=a.message,
                    detail=a.detail,
                    recommendation=a.recommendation,
                    risk_score=a.risk_score,
                ) for a in alerts
            ]

        # 4. Predicciones (si hay modelo)
        if request.include_predictions and _predictor is not None:
            seq_len = LSTM_CONFIG["sequence_length"]
            if len(df) >= seq_len:
                normalized = _preprocessor.normalize(df, fit=False)
                last_seq = normalized[-seq_len:]
                preds = _predictor.predict_multistep(
                    last_seq, steps=6, return_original_scale=True
                )
                last_ts = df.index[-1]
                for i, pred in enumerate(preds):
                    prediction_results.append(PredictionResult(
                        timestamp=(last_ts + pd.Timedelta("1h") * (i + 1)).isoformat(),
                        temperature=round(float(pred[0]), 2),
                        humidity=round(float(pred[1]), 2) if len(pred) > 1 else 0.0,
                        co2=round(float(pred[2]), 2) if len(pred) > 2 else 0.0,
                    ))

        return AnalysisResponse(
            status="success",
            risk=risk_result,
            anomalies=anomaly_results,
            alerts=alert_results,
            predictions=prediction_results,
            summary={
                "total_readings": len(df),
                "time_range": f"{df.index[0]} → {df.index[-1]}",
                "total_anomalies": len(anomaly_results),
                "total_alerts": len(alert_results),
            },
        )

    except Exception as e:
        logger.error(f"Error en análisis: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
