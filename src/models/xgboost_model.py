"""
SMISIA — Modelo XGBoost (Fase A)
Clasificador multi-clase para estado de silobolsa.
"""
import logging
import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger("smisia.xgboost")

CLASS_NAMES = ["bien", "tolerable", "problema", "critico"]
LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}
REVERSE_MAP = {i: name for i, name in enumerate(CLASS_NAMES)}


def get_feature_columns(df: pd.DataFrame) -> list:
    """Obtiene las columnas de features (excluye metadatos)."""
    exclude = {
        "silo_id", "timestamp", "label", "heuristic_label",
        "label_source", "fill_date", "imputed", "needs_review",
        "model_confidence",
    }
    return [c for c in df.columns if c not in exclude and df[c].dtype in [
        np.float64, np.float32, np.int64, np.int32, float, int
    ]]


def compute_class_weights(y: np.ndarray, n_classes: int = 4) -> dict:
    """Calcula pesos inversos a la frecuencia de cada clase."""
    counts = np.bincount(y, minlength=n_classes)
    total = len(y)
    weights = {}
    for i in range(n_classes):
        if counts[i] > 0:
            weights[i] = total / (n_classes * counts[i])
        else:
            weights[i] = 1.0
    return weights


def train_xgboost(
    df: pd.DataFrame,
    config: dict,
    label_col: str = "label",
) -> dict:
    """
    Entrena modelo XGBoost con TimeSeriesSplit.
    Retorna dict con modelo, métricas, feature importances, etc.
    """
    xgb_cfg = config["xgboost"]
    feature_cols = get_feature_columns(df)

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Samples: {len(df):,}")

    # Preparar datos
    X = df[feature_cols].values.astype(np.float32)
    y_text = df[label_col].values
    y = np.array([LABEL_MAP.get(str(v), 0) for v in y_text])

    # Reemplazar NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Class weights como sample_weight
    class_weights = compute_class_weights(y, n_classes=xgb_cfg["num_class"])
    sample_weights = np.array([class_weights[int(yi)] for yi in y])

    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=xgb_cfg["n_cv_folds"])
    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        sw_train = sample_weights[train_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sw_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            "objective": xgb_cfg["objective"],
            "num_class": xgb_cfg["num_class"],
            "max_depth": xgb_cfg["max_depth"],
            "learning_rate": xgb_cfg["learning_rate"],
            "subsample": xgb_cfg["subsample"],
            "colsample_bytree": xgb_cfg["colsample_bytree"],
            "seed": config["project"]["random_seed"],
            "verbosity": 0,
            "eval_metric": "mlogloss",
        }

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=xgb_cfg["n_estimators"],
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
            verbose_eval=False,
        )

        # Predicciones
        y_pred_proba = model.predict(dval)
        y_pred = y_pred_proba.argmax(axis=1)

        fold_f1 = f1_score(y_val, y_pred, average="macro")
        fold_report = classification_report(
            y_val, y_pred, target_names=CLASS_NAMES, output_dict=True
        )

        cv_results.append({
            "fold": fold,
            "macro_f1": fold_f1,
            "report": fold_report,
            "best_iteration": model.best_iteration,
        })
        logger.info(f"Fold {fold}: macro_F1 = {fold_f1:.4f}")

    # Entrenar modelo final con todos los datos
    logger.info("Entrenando modelo final con todos los datos...")
    dtrain_full = xgb.DMatrix(X, label=y, weight=sample_weights)

    final_model = xgb.train(
        params,
        dtrain_full,
        num_boost_round=max(r["best_iteration"] for r in cv_results) + 50,
        verbose_eval=False,
    )

    # Feature importances
    importance = final_model.get_score(importance_type="gain")
    feat_importance = {
        feature_cols[int(k.replace("f", ""))]: v
        for k, v in importance.items()
        if k.startswith("f") and k[1:].isdigit()
    }

    # Métricas finales en todo el set
    y_pred_proba_final = final_model.predict(dtrain_full)
    y_pred_final = y_pred_proba_final.argmax(axis=1)
    final_report = classification_report(
        y, y_pred_final, target_names=CLASS_NAMES, output_dict=True
    )

    avg_f1 = np.mean([r["macro_f1"] for r in cv_results])
    logger.info(f"CV promedio macro_F1: {avg_f1:.4f}")
    logger.info(f"Recall 'critico': {final_report['critico']['recall']:.4f}")

    return {
        "model": final_model,
        "feature_columns": feature_cols,
        "cv_results": cv_results,
        "final_report": final_report,
        "feature_importance": feat_importance,
        "class_names": CLASS_NAMES,
        "label_map": LABEL_MAP,
    }


def train_bootstrap_ensemble(
    df: pd.DataFrame,
    config: dict,
    label_col: str = "label",
) -> list:
    """
    Entrena N modelos bootstrap para estimación de incertidumbre.
    """
    n_models = config["xgboost"]["n_bootstrap_models"]
    seed = config["project"]["random_seed"]
    rng = np.random.default_rng(seed)
    models = []

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_text = df[label_col].values
    y = np.array([LABEL_MAP.get(str(v), 0) for v in y_text])
    class_weights = compute_class_weights(y)
    sample_weights = np.array([class_weights[int(yi)] for yi in y])

    xgb_cfg = config["xgboost"]
    params = {
        "objective": xgb_cfg["objective"],
        "num_class": xgb_cfg["num_class"],
        "max_depth": xgb_cfg["max_depth"],
        "learning_rate": xgb_cfg["learning_rate"],
        "subsample": xgb_cfg["subsample"],
        "colsample_bytree": xgb_cfg["colsample_bytree"],
        "verbosity": 0,
        "eval_metric": "mlogloss",
    }

    for i in range(n_models):
        bootstrap_idx = rng.choice(len(X), size=len(X), replace=True)
        X_boot = X[bootstrap_idx]
        y_boot = y[bootstrap_idx]
        sw_boot = sample_weights[bootstrap_idx]

        params["seed"] = seed + i
        dtrain = xgb.DMatrix(X_boot, label=y_boot, weight=sw_boot)

        model = xgb.train(
            params, dtrain,
            num_boost_round=xgb_cfg["n_estimators"] // 2,
            verbose_eval=False,
        )
        models.append(model)
        logger.info(f"Bootstrap modelo {i+1}/{n_models} entrenado")

    return models


def predict_with_uncertainty(
    models: list,
    X: np.ndarray,
    feature_cols: list,
) -> dict:
    """
    Predice usando ensemble de modelos bootstrap.
    Devuelve probabilidades medias y desviación estándar.
    """
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    dmatrix = xgb.DMatrix(X, feature_names=feature_cols)
    all_probs = []

    for model in models:
        probs = model.predict(dmatrix)
        all_probs.append(probs)

    all_probs = np.array(all_probs)  # (n_models, n_samples, n_classes)
    mean_probs = all_probs.mean(axis=0)
    std_probs = all_probs.std(axis=0)

    predicted_class = mean_probs.argmax(axis=1)
    confidence = mean_probs.max(axis=1)
    uncertainty = std_probs[np.arange(len(predicted_class)), predicted_class]

    return {
        "predicted_class": predicted_class,
        "predicted_label": [REVERSE_MAP[c] for c in predicted_class],
        "probabilities": mean_probs,
        "confidence": confidence,
        "uncertainty_std": uncertainty,
        "raw_scores": {
            name: mean_probs[:, i].tolist()
            for i, name in enumerate(CLASS_NAMES)
        },
    }


def save_model(result: dict, models_dir: str = "models"):
    """Guarda modelo, features e importances."""
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(result["model"], os.path.join(models_dir, "xgboost_model.joblib"))
    joblib.dump(
        result["feature_columns"],
        os.path.join(models_dir, "feature_columns.joblib"),
    )
    joblib.dump(
        result["feature_importance"],
        os.path.join(models_dir, "feature_importance.joblib"),
    )
    logger.info(f"Modelo XGBoost guardado en {models_dir}/")


def load_model(models_dir: str = "models") -> dict:
    """Carga modelo entrenado."""
    model = joblib.load(os.path.join(models_dir, "xgboost_model.joblib"))
    feature_cols = joblib.load(os.path.join(models_dir, "feature_columns.joblib"))
    importance = joblib.load(os.path.join(models_dir, "feature_importance.joblib"))
    return {
        "model": model,
        "feature_columns": feature_cols,
        "feature_importance": importance,
    }
