"""
SMISIA — Calibración de Probabilidades y Detección de Drift
"""

import logging
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import os

logger = logging.getLogger("smisia.calibration")


class XGBWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper para XGBoost compatible con sklearn CalibratedClassifierCV."""

    def __init__(self, model, feature_cols, n_classes=4):
        self.model = model
        self.feature_cols = feature_cols
        self.n_classes = n_classes
        self.classes_ = np.arange(n_classes)

    def fit(self, X, y):
        return self

    def predict(self, X):
        import xgboost as xgb

        dmatrix = xgb.DMatrix(X)
        probs = self.model.predict(dmatrix)
        return probs.argmax(axis=1)

    def predict_proba(self, X):
        import xgboost as xgb

        dmatrix = xgb.DMatrix(X)
        return self.model.predict(dmatrix)


def calibrate_probabilities(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: list,
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    """
    Calibra las probabilidades del modelo usando isotonic o Platt scaling.
    """
    wrapper = XGBWrapper(model, feature_cols)
    calibrated = CalibratedClassifierCV(wrapper, method=method, cv="prefit")
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    calibrated.fit(X_val, y_val)
    logger.info(f"Calibración {method} completada")
    return calibrated


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calcula Population Stability Index (PSI) para detección de drift.
    PSI > 0.25 indica drift significativo.
    """
    eps = 1e-6

    # Binning basado en la distribución de referencia
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    ref_pct = ref_counts / (ref_counts.sum() + eps)
    cur_pct = cur_counts / (cur_counts.sum() + eps)

    # Evitar log(0)
    ref_pct = np.clip(ref_pct, eps, 1.0)
    cur_pct = np.clip(cur_pct, eps, 1.0)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def check_drift(
    reference_predictions: np.ndarray,
    current_predictions: np.ndarray,
    threshold: float = 0.25,
) -> dict:
    """
    Verifica drift comparando distribuciones de predicciones.
    """
    results = {}

    if reference_predictions.ndim == 1:
        psi = compute_psi(reference_predictions, current_predictions)
        results["overall_psi"] = psi
        results["drift_detected"] = psi > threshold
    else:
        # PSI por clase
        for i in range(reference_predictions.shape[1]):
            psi = compute_psi(reference_predictions[:, i], current_predictions[:, i])
            results[f"class_{i}_psi"] = psi

        avg_psi = np.mean([v for k, v in results.items() if "psi" in k])
        results["avg_psi"] = avg_psi
        results["drift_detected"] = avg_psi > threshold

    if results["drift_detected"]:
        logger.warning(
            f"⚠️ Drift detectado! PSI: {results.get('avg_psi', results.get('overall_psi', 0)):.4f}"
        )
    else:
        logger.info("Sin drift significativo")

    return results


def save_calibration(calibrated_model, models_dir: str = "models"):
    """Guarda modelo calibrado."""
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(
        calibrated_model,
        os.path.join(models_dir, "calibrated_model.joblib"),
    )
    logger.info(f"Modelo calibrado guardado en {models_dir}/")
