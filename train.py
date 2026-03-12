"""
SMISIA — Pipeline de Entrenamiento
Ejecuta: carga datos → limpieza → features → entrenamiento de modelos → guardado.
"""
import argparse
import logging
import sys
import os
import numpy as np
import pandas as pd

# Configurar path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config  # noqa: E402
from src.preprocessing.cleaner import run_preprocessing_pipeline  # noqa: E402
from src.features.engineer import run_feature_engineering  # noqa: E402
from src.labeling.heuristic import apply_heuristic_labels, encode_labels  # noqa: E402
from src.models.xgboost_model import (  # noqa: E402
    train_xgboost, train_bootstrap_ensemble,
    save_model, get_feature_columns,
)
from src.models.lstm_model import train_lstm, save_lstm_model  # noqa: E402
from src.models.anomaly import train_anomaly_detector, save_anomaly_model  # noqa: E402
from src.models.calibration import save_calibration  # noqa: E402

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("smisia.train")


def main():
    parser = argparse.ArgumentParser(description="SMISIA — Pipeline de Entrenamiento")
    parser.add_argument("--config", type=str, default=None, help="Ruta a config.yml")
    parser.add_argument("--skip-lstm", action="store_true", help="Saltar LSTM")
    parser.add_argument("--skip-anomaly", action="store_true", help="Saltar anomaly detection")
    args = parser.parse_args()

    config = load_config(args.config)
    models_dir = config["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    seed = config["project"]["random_seed"]
    np.random.seed(seed)

    # ---------------------------------------------------------------
    # 1. Cargar datos
    # ---------------------------------------------------------------
    data_path = config["paths"]["raw_dataset"]
    logger.info(f"Cargando datos de {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Cargados {len(df):,} registros de {df['silo_id'].nunique()} silos")

    # ---------------------------------------------------------------
    # 2. Preprocesamiento
    # ---------------------------------------------------------------
    logger.info("Ejecutando preprocesamiento...")
    df = run_preprocessing_pipeline(df)

    # ---------------------------------------------------------------
    # 3. Feature Engineering
    # ---------------------------------------------------------------
    logger.info("Ejecutando feature engineering...")
    df = run_feature_engineering(df)

    # ---------------------------------------------------------------
    # 4. Labels
    # ---------------------------------------------------------------
    if "label" not in df.columns or df["label"].isna().all():
        logger.info("Sin labels reales; generando labels heurísticos...")
        df = apply_heuristic_labels(df)
        df["label"] = df["heuristic_label"]

    # Guardar dataset procesado
    processed_path = config["paths"].get("feature_dataset", "data/feature_dataset.parquet")
    os.makedirs(os.path.dirname(processed_path) or ".", exist_ok=True)
    df.to_parquet(processed_path, index=False)
    logger.info(f"Dataset procesado guardado en {processed_path}")

    # ---------------------------------------------------------------
    # 5. Fase A: XGBoost
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("FASE A: Entrenamiento XGBoost")
    logger.info("=" * 60)

    xgb_result = train_xgboost(df, config, label_col="label")
    save_model(xgb_result, models_dir)

    # Ensemble para incertidumbre
    logger.info("Entrenando ensemble bootstrap...")
    bootstrap_models = train_bootstrap_ensemble(df, config, label_col="label")
    import joblib
    joblib.dump(bootstrap_models, os.path.join(models_dir, "bootstrap_models.joblib"))
    logger.info(f"Ensemble de {len(bootstrap_models)} modelos guardado")

    # Feature columns para otros modelos
    feature_cols = get_feature_columns(df)

    # ---------------------------------------------------------------
    # 6. Fase B: LSTM
    # ---------------------------------------------------------------
    if not args.skip_lstm:
        logger.info("=" * 60)
        logger.info("FASE B: Entrenamiento LSTM")
        logger.info("=" * 60)

        # Features básicas para LSTM (no las rolling, solo raw)
        lstm_features = [
            c for c in ["temperature_c", "humidity_pct", "co2_ppm",
                        "nh3_ppm", "battery_pct"]
            if c in df.columns
        ]
        lstm_result = train_lstm(df, lstm_features, config)
        save_lstm_model(lstm_result, models_dir)
    else:
        logger.info("FASE B: LSTM omitido (--skip-lstm)")

    # ---------------------------------------------------------------
    # 7. Fase C: Anomaly Detection
    # ---------------------------------------------------------------
    if not args.skip_anomaly:
        logger.info("=" * 60)
        logger.info("FASE C: Detección de Anomalías")
        logger.info("=" * 60)

        anomaly_result = train_anomaly_detector(df, feature_cols, config)
        save_anomaly_model(anomaly_result, models_dir)
    else:
        logger.info("FASE C: Anomaly detection omitido (--skip-anomaly)")

    # ---------------------------------------------------------------
    # Resumen
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO COMPLETO")
    logger.info("=" * 60)
    logger.info(f"Modelos guardados en: {models_dir}/")

    if xgb_result.get("final_report"):
        report = xgb_result["final_report"]
        logger.info(f"  XGBoost macro F1: {report.get('macro avg', {}).get('f1-score', 'N/A'):.4f}")
        logger.info(f"  Recall 'critico': {report.get('critico', {}).get('recall', 'N/A'):.4f}")

    files = os.listdir(models_dir)
    for f in sorted(files):
        fpath = os.path.join(models_dir, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        logger.info(f"  {f}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
