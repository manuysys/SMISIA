"""
SMISIA — Pipeline de Evaluación
Genera reportes de métricas, confusion matrix, robustness tests.
"""
import argparse
import logging
import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config  # noqa: E402
from src.models.xgboost_model import (  # noqa: E402
    load_model, CLASS_NAMES, LABEL_MAP,
    predict_with_uncertainty,
)
from src.models.calibration import compute_psi  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("smisia.eval")


def evaluate_classification(df, model_data, config):
    """Evaluación completa del clasificador."""
    import xgboost as xgb
    import joblib

    feature_cols = model_data["feature_columns"]
    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_true = np.array([LABEL_MAP.get(str(v), 0) for v in df["label"].values])

    dmatrix = xgb.DMatrix(X)
    y_proba = model_data["model"].predict(dmatrix)
    y_pred = y_proba.argmax(axis=1)

    # Incertidumbre (si hay bootstrap models)
    uncertainty_stats = {}
    models_dir = config["paths"]["models_dir"]
    boot_path = os.path.join(models_dir, "bootstrap_models.joblib")
    if os.path.exists(boot_path):
        logger.info("Calculando incertidumbre con modelos bootstrap...")
        boot_models = joblib.load(boot_path)
        uncert_res = predict_with_uncertainty(boot_models, X, feature_cols)
        uncertainty_stats = {
            "mean_std": float(uncert_res["uncertainty_std"].mean()),
            "max_std": float(uncert_res["uncertainty_std"].max()),
        }

    # PSI (Estabilidad de features - simplificado como drift interno)
    # Comparamos la primera mitad vs la segunda mitad del dataset de evaluación
    mid = len(X) // 2
    psi_scores = {}
    if mid > 0:
        for i, col in enumerate(feature_cols):
            psi = compute_psi(X[:mid, i], X[mid:, i])
            if psi > 0.1:  # Solo reportar si hay algo de inestabilidad
                psi_scores[col] = round(psi, 4)

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
    )
    report_str = classification_report(y_true, y_pred, target_names=CLASS_NAMES)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # ROC AUC por clase (one-vs-rest)
    auc_scores = {}
    for i, name in enumerate(CLASS_NAMES):
        try:
            y_binary = (y_true == i).astype(int)
            if y_binary.sum() > 0 and (1 - y_binary).sum() > 0:
                auc = roc_auc_score(y_binary, y_proba[:, i])
                auc_scores[name] = round(auc, 4)
        except Exception:
            auc_scores[name] = None

    return {
        "report": report,
        "report_str": report_str,
        "confusion_matrix": cm.tolist(),
        "auc_scores": auc_scores,
        "macro_f1": report["macro avg"]["f1-score"],
        "recall_critico": report.get("critico", {}).get("recall", 0),
        "uncertainty": uncertainty_stats,
        "feature_psi": psi_scores,
    }


def robustness_test(df, model_data, noise_levels=None, missing_levels=None):
    """
    Tests de robustez: inyección de ruido y missingness.
    """
    import xgboost as xgb

    if noise_levels is None:
        noise_levels = [0.05, 0.10, 0.20]
    if missing_levels is None:
        missing_levels = [0.10, 0.30, 0.50]

    feature_cols = model_data["feature_columns"]
    X_orig = df[feature_cols].values.astype(np.float32)
    X_orig = np.nan_to_num(X_orig, nan=0.0, posinf=0.0, neginf=0.0)
    y_true = np.array([LABEL_MAP.get(str(v), 0) for v in df["label"].values])

    rng = np.random.default_rng(42)
    results = {"noise": {}, "missing": {}}

    # Test de ruido
    for noise_pct in noise_levels:
        X_noisy = X_orig.copy()
        std = X_orig.std(axis=0)
        noise = rng.normal(0, noise_pct, X_orig.shape) * std
        X_noisy += noise

        dmatrix = xgb.DMatrix(X_noisy)
        y_pred = model_data["model"].predict(dmatrix).argmax(axis=1)
        f1 = f1_score(y_true, y_pred, average="macro")
        results["noise"][f"{int(noise_pct*100)}%"] = round(f1, 4)

    # Test de missing data
    for miss_pct in missing_levels:
        X_miss = X_orig.copy()
        mask = rng.random(X_orig.shape) < miss_pct
        X_miss[mask] = 0.0

        dmatrix = xgb.DMatrix(X_miss)
        y_pred = model_data["model"].predict(dmatrix).argmax(axis=1)
        f1 = f1_score(y_true, y_pred, average="macro")
        results["missing"][f"{int(miss_pct*100)}%"] = round(f1, 4)

    return results


def main():
    parser = argparse.ArgumentParser(description="SMISIA — Evaluación")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    models_dir = config["paths"]["models_dir"]

    # Cargar datos procesados
    feature_path = config["paths"].get("feature_dataset", "data/feature_dataset.parquet")
    logger.info(f"Cargando datos de {feature_path}...")
    df = pd.read_parquet(feature_path)

    # Cargar modelo
    model_data = load_model(models_dir)

    # ---------------------------------------------------------------
    # Evaluación de clasificación
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("EVALUACIÓN DE CLASIFICACIÓN")
    logger.info("=" * 60)

    eval_results = evaluate_classification(df, model_data, config)
    print("\n" + eval_results["report_str"])
    print("\nConfusion Matrix:")
    print(np.array(eval_results["confusion_matrix"]))
    print(f"\nAUC por clase: {json.dumps(eval_results['auc_scores'], indent=2)}")
    print(f"Macro F1: {eval_results['macro_f1']:.4f}")
    print(f"Recall 'critico': {eval_results['recall_critico']:.4f}")

    if eval_results["uncertainty"]:
        print(f"Incertidumbre (Bootstrap std): Media={eval_results['uncertainty']['mean_std']:.4f}, "
              f"Max={eval_results['uncertainty']['max_std']:.4f}")

    if eval_results["feature_psi"]:
        print("\nEstabilidad de Features (PSI > 0.1):")
        for col, psi in eval_results["feature_psi"].items():
            print(f"  {col}: {psi}")

    # ---------------------------------------------------------------
    # Tests de robustez
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("TESTS DE ROBUSTEZ")
    logger.info("=" * 60)

    robust = robustness_test(df, model_data)
    print("\nDegradación con ruido (macro F1):")
    for level, f1 in robust["noise"].items():
        print(f"  Ruido {level}: F1 = {f1}")
    print("\nDegradación con missing (macro F1):")
    for level, f1 in robust["missing"].items():
        print(f"  Missing {level}: F1 = {f1}")

    # Guardar reporte
    report_path = os.path.join(models_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "classification": {
                "macro_f1": eval_results["macro_f1"],
                "recall_critico": eval_results["recall_critico"],
                "auc_scores": eval_results["auc_scores"],
                "confusion_matrix": eval_results["confusion_matrix"],
                "uncertainty": eval_results["uncertainty"],
            },
            "stability": {
                "feature_psi": eval_results["feature_psi"],
            },
            "robustness": robust,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Reporte guardado en {report_path}")


if __name__ == "__main__":
    main()
