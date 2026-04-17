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
    IngestRequest,
    IngestResponse,
    ChatRequest,
    ChatResponse,
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
from src.services.ai_service import AIService, InMemoryRepository, SensorReading as AISensorReading
from src.chatbot import ChatbotService, LLMConfig, FallbackClient

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
_ai_service: Optional[AIService] = None
_repository = InMemoryRepository()
_chatbot: Optional[ChatbotService] = None


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

    # Inicializar AI Service
    global _ai_service
    if _model is not None:
        _ai_service = AIService(
            preprocessor=_preprocessor,
            lstm_model=_model,
            risk_engine=_risk_engine,
            alert_system=_alert_system,
            repository=_repository,
            anomaly_detector=_anomaly_detector,
        )
        logger.info("✅ AI Service inicializado para API")

    # Inicializar Chatbot Service
    global _chatbot
    try:
        _chatbot = ChatbotService(ai_service=_ai_service, risk_engine=_risk_engine)
        logger.info("✅ Chatbot service inicializado")
    except Exception as e:
        logger.warning(f"⚠️ Chatbot init failed: {e}. Chat endpoint will use fallback.")
        _chatbot = ChatbotService(llm_client=FallbackClient(LLMConfig()))


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


@router.post("/ingest", response_model=IngestResponse, tags=["Tiempo Real"])
async def ingest_sensor_reading(request: IngestRequest):
    """
    Endpoint de ingestión de sensores en tiempo real.

    Recibe una nueva lectura, ejecuta el pipeline completo de ML inmediatamente,
    y devuelve el score de riesgo, predicciones, anomalías y alertas.
    """
    timestamp = request.timestamp or datetime.now().isoformat()

    reading = AISensorReading(
        silo_id=request.silo_id,
        timestamp=timestamp,
        temperature=request.temperature,
        humidity=request.humidity,
        co2=request.co2,
    )

    if _ai_service is None:
        # Fallback: usar motor de riesgo directamente sin LSTM
        sensor_vals = {
            "temperature": request.temperature,
            "humidity": request.humidity,
            "co2": request.co2,
        }
        risk_factors = _risk_engine.get_risk_factors(sensor_vals)
        alerts = _alert_system.generate_alerts(risk_factors)
        return IngestResponse(
            status="partial",  # no hay LSTM disponible
            silo_id=request.silo_id,
            timestamp=timestamp,
            risk_score=risk_factors["total_score"],
            risk_level=risk_factors["level"],
            predictions={},
            anomalies={},
            alerts=[{"level": a.level, "message": a.message} for a in alerts],
            metadata={"note": "LSTM no disponible, resultado solo con reglas de riesgo"},
        )

    try:
        result = _ai_service.ingest_and_analyze(reading)
        return IngestResponse(
            status="ok",
            silo_id=result.silo_id,
            timestamp=result.timestamp,
            risk_score=result.risk_score,
            risk_level=result.risk_level,
            predictions=result.predictions,
            anomalies=result.anomalies,
            alerts=result.alerts,
            metadata=result.metadata,
        )
    except Exception as e:
        logger.error(f"Error en /ingest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================================
# CHATBOT ENDPOINTS
# ===========================================================================

@router.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
async def chat(request: ChatRequest):
    """
    Intelligent chatbot endpoint.

    Answers questions about the silo state, alerts, sensor data,
    and agricultural risk using a real LLM API with injected system context.

    Set env var OPENROUTER_API_KEY (free at openrouter.ai) to enable.
    """
    if _chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    result = _chatbot.chat(
        message=request.message,
        silo_id=request.silo_id,
        session_id=request.session_id,
    )
    return ChatResponse(**result.to_dict())


@router.post("/chat/summarize/{silo_id}", response_model=ChatResponse, tags=["Chatbot"])
async def summarize_silo(silo_id: str):
    """
    Generate a natural-language status summary for a silo.
    """
    if _chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    result = _chatbot.summarize_silo(silo_id)
    return ChatResponse(**result.to_dict())


@router.post("/chat/explain-alert", response_model=ChatResponse, tags=["Chatbot"])
async def explain_alert(alert: dict, silo_id: str = "SILO_001"):
    """
    Get a natural-language explanation of a specific alert.
    """
    if _chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    result = _chatbot.explain_alert(alert, silo_id=silo_id)
    return ChatResponse(**result.to_dict())


@router.delete("/chat/session/{session_id}", tags=["Chatbot"])
async def clear_chat_session(session_id: str):
    """Clear conversation memory for a session."""
    if _chatbot:
        _chatbot.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}
