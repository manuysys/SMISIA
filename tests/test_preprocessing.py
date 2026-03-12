"""
SMISIA — Tests de Preprocesamiento
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_sample_df(n=100):
    """Crea DataFrame de prueba."""
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
    })


class TestValidators:
    def test_validate_schema_success(self):
        from src.preprocessing.validators import validate_schema
        df = make_sample_df()
        result = validate_schema(df)
        assert len(result) == len(df)

    def test_validate_schema_missing_column(self):
        from src.preprocessing.validators import validate_schema
        df = make_sample_df().drop(columns=["temperature_c"])
        with pytest.raises(ValueError, match="Columnas faltantes"):
            validate_schema(df)

    def test_validate_timestamps(self):
        from src.preprocessing.validators import validate_timestamps
        df = make_sample_df()
        df.loc[0, "timestamp"] = "not-a-date"
        result = validate_timestamps(df)
        assert len(result) == len(df) - 1

    def test_filter_physical_ranges(self):
        from src.preprocessing.validators import filter_physical_ranges
        df = make_sample_df()
        df.loc[0, "temperature_c"] = 999.0  # fuera de rango
        df.loc[1, "humidity_pct"] = -5.0     # fuera de rango
        result = filter_physical_ranges(df)
        assert pd.isna(result.loc[0, "temperature_c"])
        assert pd.isna(result.loc[1, "humidity_pct"])


class TestCleaner:
    def test_impute_small_gaps(self):
        from src.preprocessing.cleaner import impute_gaps
        df = make_sample_df(50)
        # Crear gap pequeño (4h = 2 lecturas)
        df.loc[10, "temperature_c"] = np.nan
        df.loc[11, "temperature_c"] = np.nan
        result = impute_gaps(df, linear_max_hours=6, ffill_max_hours=48)
        assert result.loc[10, "temperature_c"] is not None
        assert not pd.isna(result.loc[10, "temperature_c"])

    def test_pipeline_runs(self):
        from src.preprocessing.cleaner import run_preprocessing_pipeline
        df = make_sample_df()
        result = run_preprocessing_pipeline(df)
        assert len(result) > 0
        assert "imputed" in result.columns
