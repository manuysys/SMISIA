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


def compute_lead_time(df, y_pred, horizons_days=3, readings_per_day=12):
    """
    Calcula el % de eventos críticos detectados con antelación.
    Lead-time: tiempo entre la primera predicción 'problema/critico' y el primer evento 'critico' real.
    """
    lead_times = []
    
    for silo_id, group in df.groupby("silo_id"):
        group = group.sort_values("timestamp")
        y_true_silo = group["label"].values
        y_pred_silo = np.array([CLASS_NAMES[p] for p in y_pred[group.index]])
        
        # Encontrar primer índice real 'critico'
        crit_indices = np.where(y_true_silo == "critico")[0]
        if len(crit_indices) == 0:
            continue
            
        first_crit = crit_indices[0]
        
        # Encontrar primera predicción de alerta antes de ese evento
        # Alerta = 'problema' o 'critico'
        alert_indices = np.where((y_pred_silo == "problema") | (y_pred_silo == "critico"))[0]
        alert_before = alert_indices[alert_indices < first_crit]
        
        if len(alert_before) > 0:
            first_alert = alert_before[0]
            lead_h = (first_crit - first_alert) * (24 / readings_per_day)
            lead_times.append(lead_h)
            
    if not lead_times:
        return 0.0
        
    # % de críticos detectados al menos N días antes
    threshold_h = horizons_days * 24
    success_rate = np.mean([lt >= threshold_h for lt in lead_times])
    return float(success_rate)


def compute_precision_at_k(y_true, y_proba, k=50):
    """
    Calcula precisión en las top-k muestras con mayor probabilidad de ser 'critico'.
    """
    crit_probs = y_proba[:, 3] # Clase 3 = critico
    top_indices = np.argsort(crit_probs)[-k:]
    
    y_true_top = y_true[top_indices]
    precision = np.mean(y_true_top == 3)
    return float(precision)


def run_shap_analysis(model, X, feature_cols, max_samples=500):
    """
    Genera importancia global basada en SHAP (TreeExplainer).
    """
    import shap
    
    # Submuestreo para velocidad
    if len(X) > max_samples:
        X_sub = X[:max_samples]
    else:
        X_sub = X
        
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub)
    
    # SHAP devuelve una lista para multiclase. 
    # Tomamos la importancia absoluta media entre todas las clases.
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
    shap_importance = {
        feature_cols[i]: float(mean_abs_shap[i]) 
        for i in range(len(feature_cols))
    }
    # Ordenar y tomar top 20
    shap_importance = dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:20])
    return shap_importance


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

    # Métricas Base
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
    )
    report_str = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    
    # Advanced Metrics (Roadmap)
    lead_time_3d = compute_lead_time(df, y_pred, horizons_days=3)
    p_at_50 = compute_precision_at_k(y_true, y_proba, k=50)

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

    # PSI (Estabilidad de features)
    from src.models.monitoring import compute_psi
    mid = len(X) // 2
    psi_scores = {}
    if mid > 0:
        for i, col in enumerate(feature_cols):
            psi = compute_psi(X[:mid, i], X[mid:, i])
            if psi > 0.1:
                psi_scores[col] = round(psi, 4)

    # SHAP Analysis
    shap_report = {}
    try:
        logger.info("Generando análisis SHAP global...")
        shap_report = run_shap_analysis(model_data["model"], X, feature_cols)
    except Exception as e:
        logger.warning(f"Error en SHAP: {e}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # ROC AUC por clase
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
        "lead_time_3d_rate": lead_time_3d,
        "precision_at_k": p_at_50,
        "uncertainty": uncertainty_stats,
        "feature_psi": psi_scores,
        "shap_importance": shap_report,
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
    print(f"Lead-Time (>= 3 días antes): {eval_results['lead_time_3d_rate']*100:.1f}%")
    print(f"Precision@50 (Alertas Top): {eval_results['precision_at_k']:.4f}")

    if eval_results["uncertainty"]:
        print(f"Incertidumbre (Bootstrap std): Media={eval_results['uncertainty']['mean_std']:.4f}, "
              f"Max={eval_results['uncertainty']['max_std']:.4f}")

    if eval_results["feature_psi"]:
        print("\nEstabilidad de Features (PSI > 0.1):")
        for col, psi in eval_results["feature_psi"].items():
            print(f"  {col}: {psi}")
            
    if eval_results["shap_importance"]:
        print("\nImportancia Global SHAP (Top 5):")
        for i, (col, imp) in enumerate(eval_results["shap_importance"].items()):
            if i >= 5: break
            print(f"  {col}: {imp:.4f}")

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
                "lead_time_3d_rate": eval_results["lead_time_3d_rate"],
                "precision_at_k": eval_results["precision_at_k"],
                "auc_scores": eval_results["auc_scores"],
                "confusion_matrix": eval_results["confusion_matrix"],
                "uncertainty": eval_results["uncertainty"],
            },
            "stability": {
                "feature_psi": eval_results["feature_psi"],
            },
            "shap": eval_results["shap_importance"],
            "robustness": robust,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Reporte guardado en {report_path}")


if __name__ == "__main__":
    main()
