"""
SMISIA — Detección de Anomalías (Fase C)
Isolation Forest entrenado sobre datos "bien".
"""
import logging
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("smisia.anomaly")


def train_anomaly_detector(
    df: pd.DataFrame,
    feature_cols: list,
    config: dict,
) -> dict:
    """
    Entrena Isolation Forest sobre datos con label "bien".
    """
    anom_cfg = config["anomaly"]
    seed = config["project"]["random_seed"]

    # Filtrar solo datos "bien"
    bien_mask = df["label"].isin(["bien"])
    df_bien = df[bien_mask]

    if len(df_bien) < 100:
        logger.warning("Insuficientes datos 'bien' para anomaly detection")
        return {"model": None, "trained": False}

    X = df_bien[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entrenar Isolation Forest
    model = IsolationForest(
        n_estimators=anom_cfg["n_estimators"],
        contamination=anom_cfg["contamination"],
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Scores en todo el dataset
    X_all = df[feature_cols].values.astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    X_all_scaled = scaler.transform(X_all)
    raw_scores = model.decision_function(X_all_scaled)

    # Normalizar scores a 0..1 (mayor = más anómalo)
    scores_normalized = 1 - (raw_scores - raw_scores.min()) / (
        raw_scores.max() - raw_scores.min() + 1e-8
    )

    logger.info(
        f"Anomaly detector entrenado sobre {len(df_bien):,} muestras 'bien'. "
        f"Score medio: {scores_normalized.mean():.4f}"
    )

    return {
        "model": model,
        "scaler": scaler,
        "trained": True,
        "threshold": anom_cfg["threshold"],
        "feature_cols": feature_cols,
    }


def predict_anomaly_score(
    model_data: dict,
    X: np.ndarray,
) -> dict:
    """
    Calcula score de anomalía para nuevas muestras.
    Returns dict con scores normalizados y flags.
    """
    if model_data.get("model") is None:
        return {"scores": np.zeros(len(X)), "is_anomaly": np.zeros(len(X), dtype=bool)}

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_scaled = model_data["scaler"].transform(X)

    raw_scores = model_data["model"].decision_function(X_scaled)

    # Normalizar
    scores = 1 - (raw_scores - raw_scores.min()) / (
        raw_scores.max() - raw_scores.min() + 1e-8
    )

    is_anomaly = scores > model_data["threshold"]

    return {
        "scores": scores,
        "is_anomaly": is_anomaly,
    }


def save_anomaly_model(result: dict, models_dir: str = "models"):
    """Guarda modelo de anomalías."""
    os.makedirs(models_dir, exist_ok=True)
    if result.get("model") is not None:
        joblib.dump(result["model"], os.path.join(models_dir, "anomaly_model.joblib"))
        joblib.dump(result["scaler"], os.path.join(models_dir, "anomaly_scaler.joblib"))
        joblib.dump(
            {"threshold": result["threshold"], "feature_cols": result["feature_cols"]},
            os.path.join(models_dir, "anomaly_metadata.joblib"),
        )
        logger.info(f"Modelo de anomalías guardado en {models_dir}/")
