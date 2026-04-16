"""
AGRILION — Tests del Pipeline
================================

Tests unitarios para validar el funcionamiento de cada módulo
del sistema AGRILION.
"""

import sys
import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Agregar proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def sample_df():
    """Genera un DataFrame de ejemplo para tests."""
    np.random.seed(42)
    n = 200
    timestamps = pd.date_range("2025-01-01", periods=n, freq="1h")

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": 25 + 5 * np.sin(np.linspace(0, 8 * np.pi, n)) + np.random.normal(0, 1, n),
        "humidity": 60 + 8 * np.cos(np.linspace(0, 8 * np.pi, n)) + np.random.normal(0, 2, n),
        "co2": 420 + 30 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 10, n),
        "silo_id": "TEST_SILO",
    })

    # Insertar unos pocos NaN
    df.loc[10, "temperature"] = np.nan
    df.loc[50, "humidity"] = np.nan

    return df


@pytest.fixture(scope="module")
def timeseries_df(sample_df):
    """DataFrame limpio como serie temporal."""
    from src.data_loader import clean_data, to_timeseries
    df = clean_data(sample_df)
    df = to_timeseries(df)
    return df


@pytest.fixture(scope="module")
def preprocessor():
    """Instancia de DataPreprocessor."""
    from src.preprocessing import DataPreprocessor
    return DataPreprocessor(sequence_length=12)  # Más corto para tests rápidos


# =============================================================================
# TEST: DATA LOADER
# =============================================================================

class TestDataLoader:
    """Tests para el módulo data_loader."""

    def test_clean_data_removes_nan(self, sample_df):
        from src.data_loader import clean_data
        df_clean = clean_data(sample_df)

        # No debe haber NaN después de limpieza
        assert df_clean[["temperature", "humidity", "co2"]].isna().sum().sum() == 0

    def test_clean_data_preserves_shape(self, sample_df):
        from src.data_loader import clean_data
        df_clean = clean_data(sample_df)

        # Debe mantener todas las filas (solo imputa, no elimina)
        assert len(df_clean) == len(sample_df)

    def test_to_timeseries_has_datetime_index(self, sample_df):
        from src.data_loader import clean_data, to_timeseries
        df_clean = clean_data(sample_df)
        df_ts = to_timeseries(df_clean)

        assert isinstance(df_ts.index, pd.DatetimeIndex)

    def test_to_timeseries_is_sorted(self, sample_df):
        from src.data_loader import clean_data, to_timeseries
        df_clean = clean_data(sample_df)
        df_ts = to_timeseries(df_clean)

        assert df_ts.index.is_monotonic_increasing


# =============================================================================
# TEST: PREPROCESSING
# =============================================================================

class TestPreprocessing:
    """Tests para el módulo preprocessing."""

    def test_normalize_minmax_range(self, timeseries_df, preprocessor):
        normalized = preprocessor.normalize(timeseries_df, fit=True)

        assert normalized.min() >= -0.01  # Pequeña tolerancia
        assert normalized.max() <= 1.01

    def test_inverse_transform_recovers_values(self, timeseries_df, preprocessor):
        normalized = preprocessor.normalize(timeseries_df, fit=True)
        reconstructed = preprocessor.inverse_transform(normalized)

        original = timeseries_df[preprocessor.feature_columns].values
        np.testing.assert_array_almost_equal(original, reconstructed, decimal=4)

    def test_create_sequences_shapes(self, timeseries_df, preprocessor):
        normalized = preprocessor.normalize(timeseries_df, fit=True)
        X, y = preprocessor.create_sequences(normalized)

        seq_len = preprocessor.sequence_length
        n_features = normalized.shape[1]
        expected_samples = len(normalized) - seq_len

        assert X.shape == (expected_samples, seq_len, n_features)
        assert y.shape == (expected_samples, n_features)

    def test_split_data_temporal_order(self, timeseries_df, preprocessor):
        normalized = preprocessor.normalize(timeseries_df, fit=True)
        X, y = preprocessor.create_sequences(normalized)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

        # Train debe ser más grande que test
        assert len(X_train) > len(X_test)
        assert len(X_train) + len(X_test) == len(X)

    def test_prepare_pipeline(self, timeseries_df, preprocessor):
        result = preprocessor.prepare_pipeline(timeseries_df)

        assert "X_train" in result
        assert "X_test" in result
        assert "n_features" in result
        assert result["n_features"] == 3  # temperature, humidity, co2


# =============================================================================
# TEST: ANOMALY DETECTION
# =============================================================================

class TestAnomalyDetection:
    """Tests para el módulo anomaly_detection."""

    def test_zscore_detection(self, timeseries_df):
        from src.anomaly_detection import AnomalyDetector
        detector = AnomalyDetector()
        result = detector.detect_zscore(timeseries_df)

        assert "temperature_zscore" in result.columns
        assert "temperature_anomaly_zscore" in result.columns

    def test_moving_average_detection(self, timeseries_df):
        from src.anomaly_detection import AnomalyDetector
        detector = AnomalyDetector()
        result = detector.detect_moving_average(timeseries_df)

        assert "temperature_anomaly_ma" in result.columns

    def test_isolation_forest_detection(self, timeseries_df):
        from src.anomaly_detection import AnomalyDetector
        detector = AnomalyDetector()
        result = detector.detect_isolation_forest(timeseries_df)

        assert "anomaly_if" in result.columns
        assert "if_score" in result.columns

    def test_combined_detection(self, timeseries_df):
        from src.anomaly_detection import AnomalyDetector
        detector = AnomalyDetector()
        result = detector.detect_all(timeseries_df)

        assert "is_anomaly" in result.columns
        assert result["is_anomaly"].dtype == bool

    def test_anomaly_summary(self, timeseries_df):
        from src.anomaly_detection import AnomalyDetector
        detector = AnomalyDetector()
        detector.detect_all(timeseries_df)
        summary = detector.get_anomaly_summary()

        assert isinstance(summary, pd.DataFrame)


# =============================================================================
# TEST: LSTM MODEL
# =============================================================================

class TestLSTMModel:
    """Tests para el módulo lstm_model."""

    def test_build_model(self):
        from src.lstm_model import AgrilionLSTM
        model = AgrilionLSTM(sequence_length=12, n_features=3)
        model.build_model()

        assert model.model is not None
        assert model.model.input_shape == (None, 12, 3)

    def test_compile_model(self):
        from src.lstm_model import AgrilionLSTM
        model = AgrilionLSTM(sequence_length=12, n_features=3)
        model.build_model()
        model.compile_model()

        # Verificar que está compilado
        assert model.model.optimizer is not None

    def test_predict_shape(self):
        from src.lstm_model import AgrilionLSTM
        model = AgrilionLSTM(sequence_length=12, n_features=3)
        model.build_model(output_dim=3)
        model.compile_model()

        X = np.random.randn(5, 12, 3).astype(np.float32)
        predictions = model.predict(X)

        assert predictions.shape == (5, 3)

    def test_train_short(self, timeseries_df, preprocessor):
        """Test rápido de entrenamiento (2 épocas)."""
        from src.lstm_model import AgrilionLSTM

        result = preprocessor.prepare_pipeline(timeseries_df)

        model = AgrilionLSTM(
            sequence_length=preprocessor.sequence_length,
            n_features=result["n_features"]
        )
        model.build_model()
        model.compile_model()

        train_result = model.train(
            result["X_train"], result["y_train"],
            epochs=2, batch_size=16, verbose=0,
        )

        assert "final_loss" in train_result
        assert train_result["total_epochs"] > 0


# =============================================================================
# TEST: RISK ENGINE
# =============================================================================

class TestRiskEngine:
    """Tests para el módulo risk_engine."""

    def test_normal_conditions(self):
        from src.risk_engine import RiskEngine
        engine = RiskEngine()

        score = engine.calculate_risk_score({
            "temperature": 22.0,
            "humidity": 55.0,
            "co2": 380.0,
        })

        assert 0 <= score <= 30, f"Score {score} debería ser NORMAL"
        assert engine.classify_risk(score) == "NORMAL"

    def test_warning_conditions(self):
        from src.risk_engine import RiskEngine
        engine = RiskEngine()

        score = engine.calculate_risk_score({
            "temperature": 32.0,
            "humidity": 75.0,
            "co2": 650.0,
        })

        assert score > 30, f"Score {score} debería ser >= WARNING"

    def test_critical_conditions(self):
        """
        Sin datos de anomalías/LSTM, solo sensores contribuyen 55% del score.
        Valores extremos deben dar al menos WARNING, y con anomalías CRITICAL.
        """
        from src.risk_engine import RiskEngine
        engine = RiskEngine()

        # Solo sensores (máx teórico ~55)
        score_sensors = engine.calculate_risk_score({
            "temperature": 38.0,
            "humidity": 90.0,
            "co2": 1000.0,
        })
        assert score_sensors >= 30, f"Score {score_sensors} debería ser >= WARNING"

        # Con anomalías y LSTM → CRITICAL
        import pandas as pd
        anomaly_flags = pd.Series([True] * 24)  # Todas anomalías
        pred_errors = np.array([10.0] * 24)     # Errores altos

        score_full = engine.calculate_risk_score(
            {"temperature": 38.0, "humidity": 90.0, "co2": 1000.0},
            anomaly_flags=anomaly_flags,
            prediction_errors=pred_errors,
            baseline_mae=1.0,
        )
        assert score_full >= 70, f"Score {score_full} debería ser CRITICAL"
        assert engine.classify_risk(score_full) == "CRITICAL"

    def test_risk_factors_structure(self):
        from src.risk_engine import RiskEngine
        engine = RiskEngine()

        factors = engine.get_risk_factors({
            "temperature": 30.0,
            "humidity": 72.0,
            "co2": 550.0,
        })

        assert "total_score" in factors
        assert "level" in factors
        assert "factors" in factors
        assert "humidity_temp" in factors["factors"]
        assert "co2" in factors["factors"]

    def test_score_bounds(self):
        from src.risk_engine import RiskEngine
        engine = RiskEngine()

        # Valores extremos
        for temp in [0, 20, 40, 55]:
            for hum in [10, 50, 80, 99]:
                for co2 in [300, 500, 900, 2000]:
                    score = engine.calculate_risk_score({
                        "temperature": temp,
                        "humidity": hum,
                        "co2": co2,
                    })
                    assert 0 <= score <= 100, f"Score {score} out of bounds"


# =============================================================================
# TEST: ALERTS
# =============================================================================

class TestAlerts:
    """Tests para el módulo alerts."""

    def test_no_alerts_for_normal(self):
        from src.risk_engine import RiskEngine
        from src.alerts import AlertSystem

        engine = RiskEngine()
        alert_system = AlertSystem()

        factors = engine.get_risk_factors({
            "temperature": 22.0,
            "humidity": 55.0,
            "co2": 380.0,
        })

        alerts = alert_system.generate_alerts(factors)
        assert len(alerts) == 0

    def test_alerts_for_critical(self):
        from src.risk_engine import RiskEngine
        from src.alerts import AlertSystem

        engine = RiskEngine()
        alert_system = AlertSystem()

        factors = engine.get_risk_factors({
            "temperature": 38.0,
            "humidity": 90.0,
            "co2": 1000.0,
        })

        alerts = alert_system.generate_alerts(factors)
        assert len(alerts) > 0
        assert any(a.level == "CRITICAL" for a in alerts)

    def test_alert_serialization(self):
        from src.alerts import Alert
        import json

        alert = Alert(
            level="WARNING",
            category="hongos",
            message="Riesgo de hongos detectado",
            risk_score=55,
        )

        data = alert.to_dict()
        assert data["level"] == "WARNING"

        # Verificar que es serializable a JSON
        json_str = json.dumps(data, default=str)
        assert len(json_str) > 0

    def test_format_report(self):
        from src.alerts import AlertSystem, Alert

        system = AlertSystem()
        alerts = [
            Alert(level="WARNING", category="hongos",
                  message="Test warning", risk_score=50),
            Alert(level="CRITICAL", category="fermentacion",
                  message="Test critical", risk_score=85),
        ]

        report = system.format_alert_report(alerts)
        assert "WARNING" in report
        assert "CRITICAL" in report
        assert "AGRILION" in report


# =============================================================================
# TEST: PREDICTOR
# =============================================================================

class TestPredictor:
    """Tests para el módulo predictor."""

    def test_evaluate_predictions(self):
        from src.predictor import Predictor
        from src.lstm_model import AgrilionLSTM
        from src.preprocessing import DataPreprocessor

        model = AgrilionLSTM(sequence_length=12, n_features=3)
        preprocessor = DataPreprocessor(sequence_length=12)
        predictor = Predictor(model, preprocessor)

        y_true = np.random.randn(50, 3)
        y_pred = y_true + np.random.randn(50, 3) * 0.1

        metrics = predictor.evaluate_predictions(y_true, y_pred)

        assert "per_feature" in metrics
        assert "global" in metrics
        assert metrics["global"]["avg_R2"] > 0.5  # Buena predicción

    def test_detect_prediction_anomalies(self):
        from src.predictor import Predictor
        from src.lstm_model import AgrilionLSTM
        from src.preprocessing import DataPreprocessor

        model = AgrilionLSTM(sequence_length=12, n_features=3)
        preprocessor = DataPreprocessor(sequence_length=12)
        predictor = Predictor(model, preprocessor)

        y_true = np.ones((50, 3)) * 25
        y_pred = np.ones((50, 3)) * 25
        # Inyectar anomalía
        y_true[20] = [45, 90, 1000]

        result = predictor.detect_prediction_anomalies(y_true, y_pred)

        assert "is_prediction_anomaly" in result.columns
        assert result["is_prediction_anomaly"].sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
