"""
SMISIA — Tests de Modelos
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestXGBoostModel:
    def test_class_weights(self):
        from src.models.xgboost_model import compute_class_weights
        y = np.array([0, 0, 0, 0, 0, 1, 1, 2, 3])
        weights = compute_class_weights(y, n_classes=4)
        assert len(weights) == 4
        # Clase 3 (menos frecuente) debería tener mayor peso
        assert weights[3] > weights[0]

    def test_get_feature_columns(self):
        import pandas as pd
        from src.models.xgboost_model import get_feature_columns
        df = pd.DataFrame({
            "silo_id": ["A"],
            "timestamp": ["2025-01-01"],
            "temperature_c": [25.0],
            "humidity_pct": [14.0],
            "label": ["bien"],
        })
        cols = get_feature_columns(df)
        assert "temperature_c" in cols
        assert "humidity_pct" in cols
        assert "silo_id" not in cols
        assert "label" not in cols


class TestCalibration:
    def test_psi_calculation(self):
        from src.models.calibration import compute_psi
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        # Misma distribución → PSI bajo
        same = rng.normal(0, 1, 1000)
        psi_same = compute_psi(ref, same)
        assert psi_same < 0.1

        # Distribución diferente → PSI alto
        different = rng.normal(2, 1, 1000)
        psi_diff = compute_psi(ref, different)
        assert psi_diff > 0.25

    def test_drift_check(self):
        from src.models.calibration import check_drift
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(0, 1, 500)
        result = check_drift(ref, cur)
        assert "drift_detected" in result
        assert isinstance(result["drift_detected"], bool)


class TestAnomalyDetection:
    def test_isolation_forest_trains(self):
        import pandas as pd
        from src.models.anomaly import train_anomaly_detector
        rng = np.random.default_rng(42)
        n = 500
        feature_cols = ["f1", "f2", "f3"]
        df = pd.DataFrame({
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
            "f3": rng.normal(0, 1, n),
            "label": ["bien"] * n,
        })
        config = {
            "anomaly": {
                "n_estimators": 50,
                "contamination": 0.05,
                "threshold": 0.5,
            },
            "project": {"random_seed": 42},
        }
        result = train_anomaly_detector(df, feature_cols, config)
        assert result["trained"] is True
        assert result["model"] is not None
