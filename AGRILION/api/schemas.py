"""
AGRILION — API Schemas (Pydantic)
===================================

Modelos de datos para validación de requests/responses
en la API REST de AGRILION.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


# =============================================================================
# REQUEST MODELS
# =============================================================================

class SensorReading(BaseModel):
    """Lectura individual de un sensor."""
    timestamp: datetime
    temperature: float = Field(..., ge=-10, le=60, description="Temperatura en °C")
    humidity: float = Field(..., ge=0, le=100, description="Humedad relativa en %")
    co2: float = Field(..., ge=200, le=5000, description="CO2 en ppm")
    silo_id: str = Field(default="SILO_001", description="ID del silobolsa")


class PredictionRequest(BaseModel):
    """Request para predicción de valores futuros."""
    readings: List[SensorReading] = Field(
        ...,
        min_length=24,
        description="Últimas 24+ lecturas de sensores (mínimo = sequence_length)"
    )
    steps: int = Field(default=6, ge=1, le=48, description="Pasos futuros a predecir")


class AnalysisRequest(BaseModel):
    """Request para análisis completo (anomalías + riesgo + alertas)."""
    readings: List[SensorReading] = Field(
        ...,
        min_length=1,
        description="Lecturas de sensores a analizar"
    )
    include_anomaly_detection: bool = Field(default=True)
    include_risk_assessment: bool = Field(default=True)
    include_predictions: bool = Field(default=True)


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class PredictionResult(BaseModel):
    """Resultado de una predicción single/multi-step."""
    timestamp: Optional[str] = None
    temperature: float
    humidity: float
    co2: float


class AnomalyResult(BaseModel):
    """Resultado de detección de anomalía para un punto."""
    timestamp: str
    is_anomaly: bool
    affected_sensors: List[str] = []
    severity: str = "NONE"
    methods_triggered: int = 0


class AlertResult(BaseModel):
    """Alerta generada por el sistema."""
    level: str
    category: str
    message: str
    detail: str = ""
    recommendation: str = ""
    risk_score: int = 0


class RiskResult(BaseModel):
    """Resultado de evaluación de riesgo."""
    total_score: int = Field(..., ge=0, le=100)
    level: str
    emoji: str
    factors: Dict = {}


class PredictionResponse(BaseModel):
    """Response para endpoint de predicción."""
    status: str = "success"
    predictions: List[PredictionResult]
    model_info: Dict = {}


class AnalysisResponse(BaseModel):
    """Response para endpoint de análisis completo."""
    status: str = "success"
    risk: Optional[RiskResult] = None
    anomalies: List[AnomalyResult] = []
    alerts: List[AlertResult] = []
    predictions: List[PredictionResult] = []
    summary: Dict = {}


class HealthResponse(BaseModel):
    """Response para health check."""
    status: str
    version: str
    model_loaded: bool
    timestamp: str


class ModelStatusResponse(BaseModel):
    """Response para estado del modelo."""
    model_loaded: bool
    model_path: str = ""
    architecture: str = ""
    n_parameters: int = 0
    last_trained: Optional[str] = None
