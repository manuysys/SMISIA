"""
SMISIA — Rutas de la API REST
Implementa: /infer, /status, /batch_infer, /explain, /metrics
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import (
    InferRequest,
    InferResponse,
    BatchInferRequest,
    BatchInferResponse,
    StatusResponse,
    ExplainResponse,
    HealthResponse,
    MetricValue,
    ExplanationItem,
    ChatRequest,
    ChatResponse,
)
from src.chatbot.bot import format_chat_response

logger = logging.getLogger("smisia.routes")
router = APIRouter()

# Estado global (se inicializa al arrancar la app)
_state = {
    "xgb_model": None,
    "bootstrap_models": None,
    "feature_columns": None,
    "feature_importance": None,
    "anomaly_model": None,
    "anomaly_scaler": None,
    "anomaly_metadata": None,
    "start_time": time.time(),
    "inference_count": 0,
    "total_latency": 0.0,
    "last_results": {},  # cache de últimos resultados por silo_id
}

# Recomendaciones por estado
RECOMMENDATIONS = {
    "bien": "Sin acción requerida. Monitoreo rutinario.",
    "tolerable": "Monitorear con mayor frecuencia. Verificar ventilación.",
    "problema": "Inspección física y ventilación si es posible. Considerar extracción parcial.",
    "critico": "⚠️ ACCIÓN URGENTE: Inspección inmediata, ventilación forzada y considerar extracción del grano.",
}

SUMMARIES = {
    "bien": "Condiciones estables dentro de parámetros normales.",
    "tolerable": "Lecturas ligeramente fuera de rango; requiere atención.",
    "problema": "Humedad y/o temperatura en aumento; CO₂ elevado.",
    "critico": "Condiciones críticas detectadas. Riesgo alto de deterioro.",
}

CLASS_NAMES = ["bien", "tolerable", "problema", "critico"]


def init_state(state_data: dict):
    """Inicializa el estado global con modelos cargados."""
    _state.update(state_data)
    _state["start_time"] = time.time()
    logger.info("Estado de rutas inicializado")


def _readings_to_dataframe(request: InferRequest) -> pd.DataFrame:
    """Convierte lecturas del request a DataFrame."""
    records = []
    for r in request.recent_readings:
        records.append(
            {
                "silo_id": request.silo_id,
                "timestamp": r.timestamp,
                "temperature_c": r.temperature_c,
                "humidity_pct": r.humidity_pct,
                "co2_ppm": r.co2_ppm,
                "nh3_ppm": r.nh3_ppm,
                "battery_pct": r.battery_pct,
                "rssi": r.rssi,
                "snr": r.snr,
            }
        )
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["fill_date"] = request.fill_date or "2025-10-01T00:00:00+00:00"
    return df.sort_values("timestamp").reset_index(drop=True)


def _run_inference(df: pd.DataFrame, request: InferRequest) -> InferResponse:
    """Ejecuta inferencia completa."""
    from src.preprocessing.cleaner import run_preprocessing_pipeline
    from src.features.engineer import run_feature_engineering

    start_time = time.time()

    # Preprocesar y engineering
    try:
        df = run_preprocessing_pipeline(df)
        df = run_feature_engineering(df)
    except Exception as e:
        logger.error(f"Error en preprocesamiento: {e}")
        raise HTTPException(status_code=500, detail=f"Error en preprocesamiento: {e}")

    # Preparar features
    feature_cols = _state.get("feature_columns", [])
    if not feature_cols:
        raise HTTPException(status_code=500, detail="Modelo no cargado correctamente")

    # Asegurar que todas las columnas existen
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Tomar la última fila
    last_row = df.iloc[-1:][feature_cols].values.astype(np.float32)
    last_row = np.nan_to_num(last_row, nan=0.0, posinf=0.0, neginf=0.0)

    # XGBoost prediction
    if _state["xgb_model"] is not None:
        dmatrix = xgb.DMatrix(last_row)
        probs = _state["xgb_model"].predict(dmatrix)[0]
    else:
        probs = np.array([0.25, 0.25, 0.25, 0.25])

    predicted_class = int(probs.argmax())
    status = CLASS_NAMES[predicted_class]
    confidence = float(probs.max())

    # Uncertainty (bootstrap ensemble)
    uncertainty = 0.0
    if _state.get("bootstrap_models"):
        all_probs = []
        for model in _state["bootstrap_models"]:
            p = model.predict(dmatrix)[0]
            all_probs.append(p)
        all_probs = np.array(all_probs)
        uncertainty = float(all_probs.std(axis=0)[predicted_class])

    # Explanations (top features by importance)
    explanations = []
    if _state.get("feature_importance"):
        sorted_features = sorted(
            _state["feature_importance"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        total_imp = sum(v for _, v in sorted_features) or 1.0
        for feat, imp in sorted_features:
            explanations.append(
                ExplanationItem(feature=feat, impact=round(imp / total_imp, 3))
            )

    # Anomaly score
    anomaly_score = None
    if _state.get("anomaly_model") and _state.get("anomaly_scaler"):
        try:
            X_scaled = _state["anomaly_scaler"].transform(last_row)
            raw_score = _state["anomaly_model"].decision_function(X_scaled)[0]
            anomaly_score = round(float(1 - (raw_score + 0.5)), 4)
        except Exception:
            pass

    # Métricas actuales
    last_reading = request.recent_readings[-1] if request.recent_readings else None
    metrics = {}
    if last_reading:
        if last_reading.temperature_c is not None:
            metrics["temperature_c"] = MetricValue(
                value=round(last_reading.temperature_c, 1), unit="°C"
            )
        if last_reading.humidity_pct is not None:
            metrics["humidity_pct"] = MetricValue(
                value=round(last_reading.humidity_pct, 1), unit="%"
            )
        if last_reading.co2_ppm is not None:
            metrics["co2_ppm"] = MetricValue(
                value=round(last_reading.co2_ppm, 1), unit="ppm"
            )
        if last_reading.battery_pct is not None:
            metrics["battery_pct"] = MetricValue(
                value=round(last_reading.battery_pct, 1), unit="%"
            )

    # Tendencias (si hay suficientes lecturas)
    trend = {}
    if len(df) >= 12:
        for col in ["humidity_pct", "temperature_c", "co2_ppm"]:
            if col in df.columns:
                delta = df[col].iloc[-1] - df[col].iloc[-12]
                trend[f"{col}_delta_24h"] = round(float(delta), 2)

    # Raw scores
    raw_scores = {name: round(float(probs[i]), 4) for i, name in enumerate(CLASS_NAMES)}

    latency = time.time() - start_time
    _state["inference_count"] += 1
    _state["total_latency"] += latency

    result = InferResponse(
        silo_id=request.silo_id,
        timestamp=request.timestamp,
        status=status,
        confidence=round(confidence, 4),
        uncertainty_std=round(uncertainty, 4),
        summary=SUMMARIES.get(status, ""),
        metrics=metrics,
        trend=trend,
        explanations=explanations,
        recommended_action=RECOMMENDATIONS.get(status, ""),
        raw_scores=raw_scores,
        anomaly_score=anomaly_score,
    )

    # Cache
    _state["last_results"][request.silo_id] = {
        **result.model_dump(),
        "cached_at": datetime.utcnow().isoformat(),
    }

    return result


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------


@router.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    """Inferencia para una silobolsa individual."""
    df = _readings_to_dataframe(request)
    if df.empty:
        raise HTTPException(status_code=400, detail="No hay lecturas válidas")
    return _run_inference(df, request)


@router.get("/status/{silo_id}", response_model=StatusResponse)
async def get_status(silo_id: str):
    """Devuelve el último estado inferido de un silo (cache)."""
    cached = _state["last_results"].get(silo_id)
    if cached:
        return StatusResponse(
            silo_id=silo_id,
            last_update=cached.get("cached_at"),
            status=cached.get("status"),
            confidence=cached.get("confidence"),
            summary=cached.get("summary"),
            raw_scores=cached.get("raw_scores"),
        )
    return StatusResponse(
        silo_id=silo_id,
        summary="Sin inferencia reciente para este silo.",
    )


@router.post("/batch_infer", response_model=BatchInferResponse)
async def batch_infer(request: BatchInferRequest):
    """Inferencia en lote para múltiples silos."""
    results = []
    for item in request.items:
        try:
            df = _readings_to_dataframe(item)
            result = _run_inference(df, item)
            results.append(result)
        except Exception as e:
            logger.error(f"Error en batch para {item.silo_id}: {e}")

    return BatchInferResponse(
        job_id=str(uuid.uuid4()),
        results=results,
        total=len(results),
    )


@router.get("/explain/{silo_id}", response_model=ExplainResponse)
async def explain(
    silo_id: str,
    timestamp: Optional[str] = Query(None, description="Timestamp ISO8601"),
):
    """Devuelve explicaciones SHAP y top drivers."""
    cached = _state["last_results"].get(silo_id)
    if not cached:
        raise HTTPException(
            status_code=404,
            detail=f"Sin datos de inferencia para silo {silo_id}. Ejecute /infer primero.",
        )

    top_drivers = [ExplanationItem(**e) for e in cached.get("explanations", [])]

    return ExplainResponse(
        silo_id=silo_id,
        timestamp=timestamp or cached.get("timestamp"),
        top_drivers=top_drivers,
    )


@router.get("/metrics", response_model=HealthResponse)
async def metrics():
    """Salud del servicio y métricas de modelo."""
    uptime = time.time() - _state["start_time"]
    avg_latency = None
    if _state["inference_count"] > 0:
        avg_latency = round(
            (_state["total_latency"] / _state["inference_count"]) * 1000, 2
        )

    return HealthResponse(
        status="ok",
        models_loaded={
            "xgboost": _state.get("xgb_model") is not None,
            "bootstrap_ensemble": bool(_state.get("bootstrap_models")),
            "anomaly_detector": _state.get("anomaly_model") is not None,
        },
        recent_inferences=_state["inference_count"],
        avg_latency_ms=avg_latency,
        uptime_seconds=round(uptime, 1),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Endpoint del chatbot."""
    cached = None
    silo_id = request.silo_id

    # Intentar extraer silo_id del mensaje si no se proporcionó
    if not silo_id:
        import re

        match = re.search(r"silo(?:bolsa)?\s+(\w+)", request.message, re.IGNORECASE)
        if match:
            silo_id = match.group(1)

    if silo_id:
        cached = _state["last_results"].get(silo_id)

    response = format_chat_response(request.message, silo_id, cached)
    return response
