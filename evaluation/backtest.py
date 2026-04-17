"""
AGRILION — Backtesting System
================================
Simulates real-time inference on historical data and evaluates model performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0
    # Alert accuracy
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    # Early warning
    avg_warning_lead_hours: float = 0.0
    n_events: int = 0


@dataclass
class BacktestReport:
    silo_id: str
    start_time: str
    end_time: str
    total_steps: int
    metrics: BacktestMetrics = field(default_factory=BacktestMetrics)
    risk_timeline: List[dict] = field(default_factory=list)
    alert_events: List[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, indent=2)


class BacktestEngine:
    """
    Simulates step-by-step inference on historical sensor data.
    
    Usage:
        engine = BacktestEngine(preprocessor, lstm_model, risk_engine)
        report = engine.run(df_historical, seq_length=24)
    """

    def __init__(self, preprocessor, lstm_model, risk_engine, alert_system=None):
        self.preprocessor = preprocessor
        self.lstm_model = lstm_model
        self.risk_engine = risk_engine
        self.alert_system = alert_system

    def run(
        self,
        df: pd.DataFrame,
        seq_length: int = 24,
        risk_threshold: int = 70,
        silo_id: str = "SILO_001",
        ground_truth_col: Optional[str] = "anomaly_label",
    ) -> BacktestReport:
        """
        Run full backtest on historical DataFrame.

        Args:
            df: Historical sensor data with DatetimeIndex
            seq_length: LSTM input window size
            risk_threshold: Score >= this triggers a CRITICAL alert
            silo_id: Identifier
            ground_truth_col: Column with true anomaly labels (optional)

        Returns:
            BacktestReport with full metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        sensor_cols = [c for c in ["temperature", "humidity", "co2"] if c in df.columns]
        
        # Normalize full dataset once
        normalized = self.preprocessor.normalize(df, fit=False)

        y_true_list, y_pred_list = [], []
        risk_timeline = []
        alert_events = []
        predicted_alerts = []  # (timestamp, step_index)
        actual_events = []     # (timestamp, step_index)

        total_steps = len(df) - seq_length
        if total_steps <= 0:
            raise ValueError(f"Not enough data: need > {seq_length} rows, got {len(df)}")

        logger.info(f"Starting backtest: {total_steps} steps, silo={silo_id}")

        for i in range(total_steps):
            window = normalized[i : i + seq_length]
            true_next = normalized[i + seq_length]
            timestamp = df.index[i + seq_length]

            # LSTM prediction
            window_batch = window.reshape(1, seq_length, len(sensor_cols))
            pred_normalized = self.lstm_model.predict(window_batch)
            pred_original = self.preprocessor.inverse_transform(pred_normalized).flatten()
            true_original = self.preprocessor.inverse_transform(
                true_next.reshape(1, -1)
            ).flatten()

            y_true_list.append(true_original)
            y_pred_list.append(pred_original)

            # Risk score for this step
            sensor_vals = {col: float(df[col].iloc[i + seq_length]) for col in sensor_cols}
            risk_factors = self.risk_engine.get_risk_factors(sensor_vals)
            score = risk_factors["total_score"]
            level = risk_factors["level"]

            step_record = {
                "timestamp": str(timestamp),
                "step": i,
                "risk_score": score,
                "risk_level": level,
                "sensor_values": sensor_vals,
            }
            risk_timeline.append(step_record)

            # Predicted alert
            if score >= risk_threshold:
                predicted_alerts.append((timestamp, i))
                if self.alert_system:
                    alerts = self.alert_system.generate_alerts(risk_factors)
                    for a in alerts:
                        alert_events.append({
                            "timestamp": str(timestamp),
                            "level": a.level,
                            "category": a.category,
                            "message": a.message,
                            "risk_score": score,
                        })

            # Ground truth events
            if ground_truth_col and ground_truth_col in df.columns:
                label = df[ground_truth_col].iloc[i + seq_length]
                if label != "normal" and not pd.isna(label):
                    actual_events.append((timestamp, i))

        # --- Compute prediction metrics ---
        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mask = y_true != 0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else 0.0

        # --- Alert accuracy (within ±3 step window) ---
        WINDOW = 3
        tp, fp, fn = 0, 0, 0
        matched_actual = set()
        lead_times = []

        for pred_ts, pred_step in predicted_alerts:
            matched = False
            for j, (act_ts, act_step) in enumerate(actual_events):
                if j not in matched_actual and abs(pred_step - act_step) <= WINDOW:
                    tp += 1
                    matched_actual.add(j)
                    matched = True
                    lead = act_step - pred_step
                    if lead > 0:
                        lead_times.append(lead)
                    break
            if not matched:
                fp += 1

        fn = len(actual_events) - len(matched_actual)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        avg_lead = float(np.mean(lead_times)) if lead_times else 0.0

        metrics = BacktestMetrics(
            mae=round(mae, 4),
            rmse=round(rmse, 4),
            mape=round(mape, 2),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            avg_warning_lead_hours=round(avg_lead, 2),
            n_events=len(actual_events),
        )

        summary = {
            "total_predictions": total_steps,
            "critical_alerts_triggered": len(predicted_alerts),
            "actual_anomaly_events": len(actual_events),
            "model_mae": mae,
            "model_rmse": rmse,
            "alert_f1": f1,
        }

        report = BacktestReport(
            silo_id=silo_id,
            start_time=str(df.index[seq_length]),
            end_time=str(df.index[-1]),
            total_steps=total_steps,
            metrics=metrics,
            risk_timeline=risk_timeline,
            alert_events=alert_events,
            summary=summary,
        )

        logger.info(
            f"Backtest complete | MAE={mae:.4f} RMSE={rmse:.4f} "
            f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}"
        )
        return report


def print_backtest_report(report: BacktestReport):
    """Print readable summary of backtest results."""
    m = report.metrics
    print(f"""
╔══════════════════════════════════════════╗
║       AGRILION — BACKTEST REPORT         ║
╠══════════════════════════════════════════╣
  Silo:    {report.silo_id}
  Period:  {report.start_time} → {report.end_time}
  Steps:   {report.total_steps}

  ── Prediction Accuracy ──
  MAE:     {m.mae}
  RMSE:    {m.rmse}
  MAPE:    {m.mape}%

  ── Alert Accuracy ──
  Events (real):   {m.n_events}
  True Positives:  {m.true_positives}
  False Positives: {m.false_positives}
  False Negatives: {m.false_negatives}
  Precision:       {m.precision:.3f}
  Recall:          {m.recall:.3f}
  F1 Score:        {m.f1:.3f}
  Avg Lead Time:   {m.avg_warning_lead_hours}h
╚══════════════════════════════════════════╝
""")
