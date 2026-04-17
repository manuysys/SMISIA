"""
AGRILION — Model Quality Reporting
=====================================
Generates structured reports from backtest results and model evaluation.
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_error_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Analyze error distribution per feature."""
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    errors = np.abs(y_true - y_pred)
    dist = {}
    for i in range(errors.shape[1]):
        e = errors[:, i]
        dist[f"feature_{i}"] = {
            "mean": round(float(np.mean(e)), 4),
            "std": round(float(np.std(e)), 4),
            "p50": round(float(np.percentile(e, 50)), 4),
            "p90": round(float(np.percentile(e, 90)), 4),
            "p99": round(float(np.percentile(e, 99)), 4),
            "max": round(float(np.max(e)), 4),
        }
    return dist


def estimate_prediction_confidence(
    y_pred: np.ndarray, error_std: float, confidence_level: float = 0.95,
) -> np.ndarray:
    """Estimate prediction confidence intervals using error std."""
    from scipy import stats
    z = stats.norm.ppf((1 + confidence_level) / 2)
    margin = z * error_std
    lower = y_pred - margin
    upper = y_pred + margin
    return np.stack([lower, upper], axis=-1)


def compute_early_warning_score(
    risk_timeline: List[dict], actual_events: List[int],
    risk_threshold: int = 70, max_lead_steps: int = 24,
) -> dict:
    """Compute how early the system detects risk before threshold breach."""
    if not actual_events or not risk_timeline:
        return {"avg_lead_steps": 0, "pct_detected_early": 0.0, "n_events": 0}

    lead_times = []
    detected = 0
    scores = {r["step"]: r["risk_score"] for r in risk_timeline}

    for event_step in actual_events:
        for lead in range(1, max_lead_steps + 1):
            check_step = event_step - lead
            if check_step in scores and scores[check_step] >= risk_threshold:
                lead_times.append(lead)
                detected += 1
                break

    return {
        "avg_lead_steps": round(float(np.mean(lead_times)), 2) if lead_times else 0.0,
        "pct_detected_early": round(detected / len(actual_events) * 100, 1),
        "n_events": len(actual_events),
        "n_detected_early": detected,
    }


class ReportGenerator:
    """Generates structured quality reports from model evaluation data."""

    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_from_backtest(self, backtest_report, y_true=None, y_pred=None, save=True) -> dict:
        """Generate full quality report from a BacktestReport object."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "silo_id": backtest_report.silo_id,
            "evaluation_period": {
                "start": backtest_report.start_time,
                "end": backtest_report.end_time,
                "total_steps": backtest_report.total_steps,
            },
            "prediction_metrics": {
                "mae": backtest_report.metrics.mae,
                "rmse": backtest_report.metrics.rmse,
                "mape_pct": backtest_report.metrics.mape,
            },
            "alert_metrics": {
                "precision": backtest_report.metrics.precision,
                "recall": backtest_report.metrics.recall,
                "f1": backtest_report.metrics.f1,
                "true_positives": backtest_report.metrics.true_positives,
                "false_positives": backtest_report.metrics.false_positives,
                "false_negatives": backtest_report.metrics.false_negatives,
                "avg_warning_lead_hours": backtest_report.metrics.avg_warning_lead_hours,
            },
            "risk_distribution": self._compute_risk_distribution(backtest_report.risk_timeline),
            "early_warning": {"avg_lead_hours": backtest_report.metrics.avg_warning_lead_hours},
        }

        if y_true is not None and y_pred is not None:
            report["error_distribution"] = compute_error_distribution(y_true, y_pred)

        report["quality_assessment"] = self._assess_quality(report)

        if save:
            path = self.output_dir / f"report_{backtest_report.silo_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved: {path}")

        return report

    def _compute_risk_distribution(self, risk_timeline: List[dict]) -> dict:
        if not risk_timeline:
            return {}
        levels = [r["risk_level"] for r in risk_timeline]
        total = len(levels)
        counts = {"NORMAL": 0, "WARNING": 0, "CRITICAL": 0}
        for l in levels:
            counts[l] = counts.get(l, 0) + 1
        return {k: {"count": v, "pct": round(v / total * 100, 1)} for k, v in counts.items()}

    def _assess_quality(self, report: dict) -> dict:
        """Simple rule-based quality assessment."""
        mae = report["prediction_metrics"]["mae"]
        f1 = report["alert_metrics"]["f1"]
        fp = report["alert_metrics"]["false_positives"]

        issues = []
        if mae > 2.0:
            issues.append("High MAE — consider retraining model")
        if f1 < 0.6:
            issues.append("Low alert F1 — tune risk thresholds")
        if fp > 10:
            issues.append("Many false positives — increase risk threshold")

        grade = ("A" if (mae < 1.0 and f1 > 0.8) else
                 "B" if (mae < 2.0 and f1 > 0.6) else
                 "C" if (mae < 3.0 and f1 > 0.4) else "D")

        return {"grade": grade, "issues": issues,
                "recommendations": issues if issues else ["System performing well"]}

    def print_summary(self, report: dict):
        """Print readable summary."""
        pm = report["prediction_metrics"]
        am = report["alert_metrics"]
        qa = report["quality_assessment"]
        issues_str = "\n".join(f"  * {i}" for i in qa["issues"]) or "  None"
        print(f"\n  AGRILION — MODEL QUALITY REPORT")
        print(f"  Grade: {qa['grade']}")
        print(f"  Prediction: MAE={pm['mae']} RMSE={pm['rmse']} MAPE={pm['mape_pct']}%")
        print(f"  Alerts: P={am['precision']:.3f} R={am['recall']:.3f} F1={am['f1']:.3f}")
        print(f"  Lead time: {am['avg_warning_lead_hours']}h avg")
        print(f"  Issues:\n{issues_str}\n")
