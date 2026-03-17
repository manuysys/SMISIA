"""
SMISIA — Optuna Hyperparameter Tuner
Búsqueda automatizada de parámetros óptimos para XGBoost.
"""

import logging
import optuna
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
from src.models.xgboost_model import get_feature_columns, LABEL_MAP, compute_class_weights

logger = logging.getLogger("smisia.tuner")

def objective(trial, X, y, sw, n_classes, n_cv_folds):
    params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "verbosity": 0,
        "seed": 42,
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
    }
    
    tscv = TimeSeriesSplit(n_splits=n_cv_folds)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        sw_train = sw[train_idx]
        
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sw_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dval, "val")],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        preds = model.predict(dval).argmax(axis=1)
        score = f1_score(y_val, preds, average="macro")
        scores.append(score)
        
    return np.mean(scores)

def run_hyperparameter_tuning(df: pd.DataFrame, config: dict, n_trials: int = 20) -> dict:
    """Ejecuta optimización con Optuna."""
    logger.info(f"Iniciando tuning con {n_trials} trials...")
    
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array([LABEL_MAP.get(str(v), 0) for v in df["label"].values])
    
    n_classes = config["xgboost"]["num_class"]
    n_cv_folds = config["xgboost"]["n_cv_folds"]
    class_weights = compute_class_weights(y, n_classes)
    sample_weights = np.array([class_weights[int(yi)] for yi in y])
    
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X, y, sample_weights, n_classes, n_cv_folds),
        n_trials=n_trials
    )
    
    logger.info(f"Tuning completado. Mejor F1: {study.best_value:.4f}")
    return study.best_params
