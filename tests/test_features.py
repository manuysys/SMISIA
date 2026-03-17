"""
SMISIA — Tests de Feature Engineering
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_processed_df(n=200):
    """Crea DataFrame preprocesado de prueba."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2025-10-01", periods=n, freq="2h", tz="UTC")
    return pd.DataFrame({
        "silo_id": "TEST_001",
        "timestamp": timestamps,
        "temperature_c": rng.normal(25, 3, n),
        "humidity_pct": rng.normal(14, 2, n),
        "co2_ppm": rng.normal(500, 50, n),
        "nh3_ppm": rng.normal(5, 1, n),
        "battery_pct": np.linspace(100, 85, n),
        "rssi": rng.integers(-100, -60, n),
        "snr": rng.normal(10, 2, n),
        "fill_date": "2025-09-20T00:00:00+00:00",
        "imputed": False,
        "label": "bien",
    })


class TestFeatureEngineering:
    def test_rolling_features_created(self):
        from src.features.engineer import compute_rolling_features
        df = make_processed_df()
        result = compute_rolling_features(df)
        # Check que se crearon features con sufijos de ventana
        feature_cols = [c for c in result.columns if "_6h_" in c or "_24h_" in c]
        assert len(feature_cols) > 0

    def test_humidity_counters(self):
        from src.features.engineer import compute_humidity_counters
        df = make_processed_df()
        result = compute_humidity_counters(df)
        assert "hours_humidity_above_16_24h" in result.columns
        assert "consecutive_hours_humidity_increase_24h" in result.columns

    def test_combined_signals(self):
        from src.features.engineer import compute_combined_signals
        df = make_processed_df()
        # Agregar columnas necesarias
        df["temperature_c_24h_slope"] = 0.1
        df["humidity_pct_24h_slope"] = 0.1
        df["co2_ppm_6h_max"] = 600
        df["co2_ppm_168h_mean"] = 500
        df["co2_ppm_168h_std"] = 20
        result = compute_combined_signals(df)
        assert "temp_and_humidity_up_24h" in result.columns
        assert "co2_spike_recent" in result.columns

    def test_static_features(self):
        from src.features.engineer import compute_static_features
        df = make_processed_df()
        result = compute_static_features(df)
        assert "days_since_fill" in result.columns
        assert result["days_since_fill"].min() >= 0

    def test_full_pipeline(self):
        from src.features.engineer import run_feature_engineering
        df = make_processed_df()
        result = run_feature_engineering(df)
        assert len(result.columns) > len(df.columns)
