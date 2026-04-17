"""
AGRILION — AI Service
========================
Orchestrates full ML pipeline: ingest → preprocess → detect → predict → score → alert.
Designed for real-time and batch operation with an abstract DB layer.
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# ─── Result Schema ────────────────────────────────────────────────────────────

@dataclass
class SensorReading:
    silo_id: str
    timestamp: str
    temperature: float
    humidity: float
    co2: float


@dataclass
class AIResult:
    silo_id: str
    timestamp: str
    risk_score: int
    risk_level: str
    risk_explanation: str
    predictions: dict
    anomalies: dict
    alerts: list
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, indent=2)


# ─── Abstract DB Repository ───────────────────────────────────────────────────

class AbstractSiloRepository:
    """
    Abstract base for database integration.
    Implement this for Firebase, PostgreSQL, InfluxDB, etc.
    """

    def get_recent_readings(self, silo_id: str, n: int = 50) -> pd.DataFrame:
        raise NotImplementedError

    def save_reading(self, reading: SensorReading) -> bool:
        raise NotImplementedError

    def save_result(self, result: AIResult) -> bool:
        raise NotImplementedError

    def get_result_history(self, silo_id: str, hours: int = 24) -> List[dict]:
        raise NotImplementedError


class InMemoryRepository(AbstractSiloRepository):
    """
    In-memory repository for testing / simulation.
    Replace with Firebase/SQL implementation in production.
    """

    def __init__(self):
        self._readings: Dict[str, List[dict]] = {}
        self._results: Dict[str, List[dict]] = {}

    def save_reading(self, reading: SensorReading) -> bool:
        sid = reading.silo_id
        if sid not in self._readings:
            self._readings[sid] = []
        self._readings[sid].append(asdict(reading))
        return True

    def get_recent_readings(self, silo_id: str, n: int = 50) -> pd.DataFrame:
        readings = self._readings.get(silo_id, [])
        if not readings:
            return pd.DataFrame()
        df = pd.DataFrame(readings[-n:])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        return df

    def save_result(self, result: AIResult) -> bool:
        sid = result.silo_id
        if sid not in self._results:
            self._results[sid] = []
        self._results[sid].append(result.to_dict())
        return True

    def get_result_history(self, silo_id: str, hours: int = 24) -> List[dict]:
        results = self._results.get(silo_id, [])
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        return [r for r in results if r["timestamp"] >= cutoff]

    def load_dataframe(self, silo_id: str, df: pd.DataFrame):
        """Seed the repository with an existing DataFrame (for testing)."""
        for ts, row in df.iterrows():
            reading = SensorReading(
                silo_id=silo_id,
                timestamp=str(ts),
                temperature=float(row.get("temperature", 0)),
                humidity=float(row.get("humidity", 0)),
                co2=float(row.get("co2", 0)),
            )
            self.save_reading(reading)


# ─── AI Service ───────────────────────────────────────────────────────────────

class AIService:
    """
    Full ML pipeline service.

    Usage:
        svc = AIService(preprocessor, lstm, risk_engine, alert_system, repo)
        result = svc.ingest_and_analyze(reading)
    """

    def __init__(
        self,
        preprocessor,
        lstm_model,
        risk_engine,
        alert_system,
        repository: AbstractSiloRepository,
        seq_length: int = 24,
        anomaly_detector=None,
    ):
        self.preprocessor = preprocessor
        self.lstm_model = lstm_model
        self.risk_engine = risk_engine
        self.alert_system = alert_system
        self.repo = repository
        self.seq_length = seq_length
        self.anomaly_detector = anomaly_detector
        self._sensor_cols = ["temperature", "humidity", "co2"]

    def ingest_and_analyze(self, reading: SensorReading) -> AIResult:
        """
        Main entry point: ingest one reading, run full pipeline, return result.

        Args:
            reading: new SensorReading from IoT device

        Returns:
            AIResult with risk score, predictions, alerts
        """
        # 1. Store reading
        self.repo.save_reading(reading)

        # 2. Fetch recent history for context
        df = self.repo.get_recent_readings(reading.silo_id, n=self.seq_length + 10)

        # 3. Ensure we have enough data
        if len(df) < self.seq_length:
            logger.warning(
                f"Only {len(df)} readings for {reading.silo_id}, "
                f"need {self.seq_length}. Returning preliminary result."
            )
            return self._preliminary_result(reading)

        # 4. Preprocess
        sensor_data = df[self._sensor_cols].dropna()
        normalized = self.preprocessor.normalize(sensor_data, fit=False)

        # 5. LSTM prediction (next step)
        last_seq = normalized[-self.seq_length:].reshape(1, self.seq_length, len(self._sensor_cols))
        pred_normalized = self.lstm_model.predict(last_seq)
        pred_original = self.preprocessor.inverse_transform(pred_normalized).flatten()

        predictions = {
            col: round(float(pred_original[i]), 2)
            for i, col in enumerate(self._sensor_cols)
        }

        # 6. Anomaly detection
        anomalies = {}
        if self.anomaly_detector and len(sensor_data) >= 12:
            try:
                anom_df = self.anomaly_detector.detect_all(sensor_data)
                last = anom_df.iloc[-1]
                anomalies = {
                    "is_anomaly": bool(last.get("is_anomaly", False)),
                    "affected_sensors": [
                        c for c in self._sensor_cols
                        if last.get(f"{c}_anomaly_consensus", False)
                    ],
                }
            except Exception as e:
                logger.warning(f"Anomaly detection skipped: {e}")

        # 7. Risk scoring
        current_vals = {
            col: float(getattr(reading, col))
            for col in self._sensor_cols
        }
        risk_factors = self.risk_engine.get_risk_factors(current_vals)
        score = risk_factors["total_score"]
        level = risk_factors["level"]

        # 8. Alert generation
        alerts_raw = self.alert_system.generate_alerts(risk_factors)
        alerts = [
            {
                "level": a.level,
                "category": a.category,
                "message": a.message,
                "recommendation": a.recommendation,
            }
            for a in alerts_raw
        ]

        result = AIResult(
            silo_id=reading.silo_id,
            timestamp=reading.timestamp,
            risk_score=score,
            risk_level=level,
            risk_explanation=risk_factors.get("factors", {}).get(
                "humidity_temp", {}
            ).get("detail", ""),
            predictions=predictions,
            anomalies=anomalies,
            alerts=alerts,
            metadata={
                "seq_length_used": self.seq_length,
                "n_readings_available": len(df),
                "pipeline_version": "1.0",
            },
        )

        # 9. Persist result
        self.repo.save_result(result)

        logger.info(
            f"[{reading.silo_id}] Score={score} Level={level} "
            f"Alerts={len(alerts)} Predictions={predictions}"
        )

        return result

    def analyze_batch(self, silo_id: str) -> Optional[AIResult]:
        """Run pipeline on latest stored data without ingesting new reading."""
        df = self.repo.get_recent_readings(silo_id, n=self.seq_length + 10)
        if df.empty or len(df) < self.seq_length:
            return None
        last_row = df.iloc[-1]
        reading = SensorReading(
            silo_id=silo_id,
            timestamp=str(df.index[-1]),
            temperature=float(last_row.get("temperature", 0)),
            humidity=float(last_row.get("humidity", 0)),
            co2=float(last_row.get("co2", 0)),
        )
        return self.ingest_and_analyze(reading)

    def _preliminary_result(self, reading: SensorReading) -> AIResult:
        """Return basic result when insufficient history is available."""
        current_vals = {col: float(getattr(reading, col)) for col in self._sensor_cols}
        risk_factors = self.risk_engine.get_risk_factors(current_vals)
        score = risk_factors["total_score"]
        return AIResult(
            silo_id=reading.silo_id,
            timestamp=reading.timestamp,
            risk_score=score,
            risk_level=risk_factors["level"],
            risk_explanation="Preliminary — insufficient history for LSTM",
            predictions={},
            anomalies={},
            alerts=[],
            metadata={"status": "preliminary"},
        )
