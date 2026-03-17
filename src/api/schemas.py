"""
SMISIA — Esquemas Pydantic para la API
"""

from pydantic import BaseModel, Field
from typing import Optional


class SensorReading(BaseModel):
    """Una lectura individual de sensor."""

    timestamp: str
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    co2_ppm: Optional[float] = None
    nh3_ppm: Optional[float] = None
    battery_pct: Optional[float] = None
    rssi: Optional[int] = None
    snr: Optional[float] = None


class InferRequest(BaseModel):
    """Request para /infer."""

    silo_id: str = Field(..., description="Identificador de la silobolsa")
    timestamp: str = Field(..., description="Timestamp ISO8601")
    recent_readings: list[SensorReading] = Field(
        ..., description="Lecturas recientes del silo"
    )
    fill_date: Optional[str] = Field(None, description="Fecha de llenado ISO8601")


class MetricValue(BaseModel):
    """Valor de una métrica con unidad."""

    value: float
    unit: str


class ExplanationItem(BaseModel):
    """Un driver explicativo."""

    feature: str
    impact: float


class InferResponse(BaseModel):
    """Respuesta de /infer."""

    silo_id: str
    timestamp: str
    status: str
    confidence: float
    uncertainty_std: float
    summary: str
    metrics: dict[str, MetricValue]
    trend: dict[str, float]
    explanations: list[ExplanationItem]
    recommended_action: str
    raw_scores: dict[str, float]
    anomaly_score: Optional[float] = None
    predictions: Optional[dict] = None


class BatchInferRequest(BaseModel):
    """Request para /batch_infer."""

    items: list[InferRequest]


class BatchInferResponse(BaseModel):
    """Respuesta de /batch_infer."""

    job_id: str
    results: list[InferResponse]
    total: int


class StatusResponse(BaseModel):
    """Respuesta de /status/{silo_id}."""

    silo_id: str
    last_update: Optional[str] = None
    status: Optional[str] = None
    confidence: Optional[float] = None
    summary: Optional[str] = None
    raw_scores: Optional[dict] = None


class ExplainResponse(BaseModel):
    """Respuesta de /explain/{silo_id}."""

    silo_id: str
    timestamp: Optional[str] = None
    top_drivers: list[ExplanationItem]
    shap_values: Optional[dict] = None


class HealthResponse(BaseModel):
    """Respuesta de /metrics (salud del servicio)."""

    status: str
    models_loaded: dict[str, bool]
    recent_inferences: int
    avg_latency_ms: Optional[float] = None
    uptime_seconds: float


class ChatRequest(BaseModel):
    """Request del chatbot."""

    message: str
    silo_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Respuesta del chatbot."""

    brief: str
    detail: Optional[str] = None
    silo_id: Optional[str] = None
