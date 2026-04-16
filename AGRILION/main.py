"""
AGRILION -- Pipeline Principal
================================

Pipeline end-to-end completo:
1. Generar datos sinteticos (si no existen)
2. Cargar y limpiar datos
3. Preprocesar (normalizar, crear secuencias)
4. Entrenar modelo LSTM
5. Detectar anomalias (3 metodos + consenso)
6. Predecir valores futuros
7. Calcular riesgo agregado
8. Generar alertas
9. Producir graficos
10. Imprimir reporte en consola

Uso:
    python main.py                           # Pipeline completo
    python main.py --skip-training           # Usar modelo existente
    python main.py --epochs 100              # Personalizar epocas
    python main.py --data path/to/data.csv   # Usar datos propios
"""

import sys
import os
import io
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

# Forzar UTF-8 en stdout/stderr para Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# Configurar logging ANTES de importar modulos
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AGRILION")

# Suprimir logs verbosos
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logging.getLogger("sklearn").setLevel(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Imports del proyecto
from src.config import (
    DEFAULT_CSV_PATH, DEFAULT_MODEL_PATH, SCALER_PATH,
    SENSOR_COLUMNS, LSTM_CONFIG, OUTPUTS_DIR,
)
from src.data_loader import load_and_prepare, load_csv, clean_data, to_timeseries
from src.preprocessing import DataPreprocessor
from src.anomaly_detection import AnomalyDetector
from src.lstm_model import AgrilionLSTM
from src.predictor import Predictor
from src.risk_engine import RiskEngine
from src.alerts import AlertSystem
from src.visualization import generate_all_plots
from data.synthetic_generator import generate_synthetic_data


def print_banner():
    """Imprime el banner de inicio."""
    banner = r"""
    ==============================================================
    |                                                            |
    |       _   ___ ___ ___ _    ___ ___  _  _                   |
    |      /_\ / __| _ \_ _| |  |_ _/ _ \| \| |                  |
    |     / _ \ (_ |   /| || |__ | | (_) | .` |                  |
    |    /_/ \_\___|_|_\___|____|___\___/|_|\_|                  |
    |                                                            |
    |    Sistema ML de Analisis de Silobolsas Agricolas          |
    |    v1.0.0                                                  |
    ==============================================================
    """
    print(banner)


def parse_args():
    """Parse argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="AGRILION — Pipeline ML para silobolsas agrícolas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Ruta al archivo CSV de datos (default: genera datos sintéticos)"
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Saltar entrenamiento y usar modelo existente"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=f"Número de épocas de entrenamiento (default: {LSTM_CONFIG['epochs']})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help=f"Tamaño de batch (default: {LSTM_CONFIG['batch_size']})"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="No generar gráficos"
    )
    parser.add_argument(
        "--show-plots", action="store_true",
        help="Mostrar gráficos en pantalla además de guardarlos"
    )
    return parser.parse_args()


def run_pipeline(args=None):
    """
    Ejecuta el pipeline completo de AGRILION.

    Parameters
    ----------
    args : argparse.Namespace, optional
        Argumentos de línea de comandos.

    Returns
    -------
    dict
        Resultados del pipeline con todos los datos generados.
    """
    if args is None:
        args = parse_args()

    start_time = time.time()
    print_banner()

    results = {}

    # =========================================================================
    # PASO 1: DATOS
    # =========================================================================
    print("\n" + "=" * 70)
    print("  📦 PASO 1/8 — CARGA DE DATOS")
    print("=" * 70)

    data_path = args.data or str(DEFAULT_CSV_PATH)

    if not Path(data_path).exists():
        logger.info("No se encontraron datos. Generando datos sintéticos...")
        df_raw = generate_synthetic_data(output_path=data_path)
    else:
        logger.info(f"Cargando datos desde: {data_path}")
        df_raw = load_csv(data_path)

    # Limpiar y convertir a serie temporal
    df_clean = clean_data(df_raw)
    df_ts = to_timeseries(df_clean)

    logger.info(f"✅ Datos preparados: {len(df_ts)} registros")
    logger.info(f"   Rango: {df_ts.index[0]} → {df_ts.index[-1]}")
    results["df_ts"] = df_ts

    # =========================================================================
    # PASO 2: PREPROCESAMIENTO
    # =========================================================================
    print("\n" + "=" * 70)
    print("  ⚙️  PASO 2/8 — PREPROCESAMIENTO")
    print("=" * 70)

    preprocessor = DataPreprocessor()
    pipeline_data = preprocessor.prepare_pipeline(df_ts)

    X_train = pipeline_data["X_train"]
    X_test = pipeline_data["X_test"]
    y_train = pipeline_data["y_train"]
    y_test = pipeline_data["y_test"]
    n_features = pipeline_data["n_features"]

    logger.info(f"✅ Preprocesamiento completado:")
    logger.info(f"   Train: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"   Test:  X={X_test.shape}, y={y_test.shape}")
    logger.info(f"   Features: {n_features}")

    results["preprocessor"] = preprocessor

    # =========================================================================
    # PASO 3: MODELO LSTM
    # =========================================================================
    print("\n" + "=" * 70)
    print("  🧠 PASO 3/8 — MODELO LSTM")
    print("=" * 70)

    lstm = AgrilionLSTM(
        sequence_length=LSTM_CONFIG["sequence_length"],
        n_features=n_features,
    )

    training_history = None

    if args.skip_training and Path(DEFAULT_MODEL_PATH).exists():
        logger.info("Cargando modelo existente...")
        lstm.load()
        preprocessor.load_scaler()
    else:
        # Construir y compilar
        lstm.build_model()
        lstm.compile_model()

        # Entrenar
        epochs = args.epochs or LSTM_CONFIG["epochs"]
        batch_size = args.batch_size or LSTM_CONFIG["batch_size"]

        train_result = lstm.train(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )
        training_history = train_result["history"]

        # Evaluar en test
        eval_metrics = lstm.evaluate(X_test, y_test)
        logger.info(f"✅ Evaluación en test: Loss={eval_metrics['test_loss']:.6f}, "
                     f"MAE={eval_metrics['test_mae']:.6f}")

        # Guardar modelo y scaler
        lstm.save()
        preprocessor.save_scaler()

    results["lstm"] = lstm
    results["training_history"] = training_history

    # =========================================================================
    # PASO 4: PREDICCIÓN
    # =========================================================================
    print("\n" + "=" * 70)
    print("  🔮 PASO 4/8 — PREDICCIÓN")
    print("=" * 70)

    predictor = Predictor(lstm, preprocessor)

    # Predecir sobre test set
    y_pred_normalized = lstm.predict(X_test)
    y_pred = preprocessor.inverse_transform(y_pred_normalized)
    y_true = preprocessor.inverse_transform(y_test)

    # Evaluar predicciones en escala original
    forecast_report = predictor.generate_forecast_report(
        y_true, y_pred,
        timestamps=df_ts.index[-len(y_test):],
        feature_names=SENSOR_COLUMNS,
    )

    logger.info(f"✅ Predicción completada")

    # Predicción multi-step (futuro)
    last_sequence = pipeline_data["normalized_data"][-LSTM_CONFIG["sequence_length"]:]
    future_predictions = predictor.predict_multistep(
        last_sequence, steps=12, return_original_scale=True
    )

    logger.info(f"   Predicción a futuro: {len(future_predictions)} pasos")
    for i, pred in enumerate(future_predictions[:3]):
        logger.info(f"      t+{i+1}: Temp={pred[0]:.1f}°C, Hum={pred[1]:.1f}%, CO2={pred[2]:.0f}ppm")
    if len(future_predictions) > 3:
        logger.info(f"      ... ({len(future_predictions) - 3} más)")

    results["y_true"] = y_true
    results["y_pred"] = y_pred
    results["forecast_report"] = forecast_report
    results["future_predictions"] = future_predictions

    # =========================================================================
    # PASO 5: DETECCIÓN DE ANOMALÍAS
    # =========================================================================
    print("\n" + "=" * 70)
    print("  🔍 PASO 5/8 — DETECCIÓN DE ANOMALÍAS")
    print("=" * 70)

    detector = AnomalyDetector()
    anomaly_df = detector.detect_all(df_ts)

    anomaly_summary = detector.get_anomaly_summary()
    if len(anomaly_summary) > 0:
        logger.info(f"\n📋 Resumen de anomalías:")
        logger.info(f"   Total puntos anómalos: {anomaly_df['is_anomaly'].sum()}")
        logger.info(f"   Porcentaje: {anomaly_df['is_anomaly'].mean() * 100:.1f}%")
    else:
        logger.info("✅ No se detectaron anomalías por consenso")

    results["anomaly_df"] = anomaly_df
    results["anomaly_summary"] = anomaly_summary

    # =========================================================================
    # PASO 6: EVALUACIÓN DE RIESGO
    # =========================================================================
    print("\n" + "=" * 70)
    print("  ⚠️  PASO 6/8 — EVALUACIÓN DE RIESGO")
    print("=" * 70)

    risk_engine = RiskEngine()

    # Calcular errores de predicción para el risk engine
    prediction_errors = None
    if y_true is not None and y_pred is not None:
        prediction_errors = np.mean(np.abs(y_true - y_pred), axis=1)

    # Evaluar riesgo en toda la timeline
    risk_df = risk_engine.evaluate_timeline(
        df_ts,
        anomaly_df=anomaly_df,
        prediction_errors=prediction_errors,
        baseline_mae=forecast_report["metrics"]["global"]["avg_MAE"] if forecast_report else None,
    )

    results["risk_df"] = risk_df

    # Riesgo actual (último punto)
    current_sensor_data = {
        col: float(df_ts[col].iloc[-1])
        for col in SENSOR_COLUMNS if col in df_ts.columns
    }
    current_risk = risk_engine.get_risk_factors(
        current_sensor_data,
        anomaly_flags=anomaly_df["is_anomaly"] if "is_anomaly" in anomaly_df.columns else None,
        prediction_errors=prediction_errors,
        baseline_mae=forecast_report["metrics"]["global"]["avg_MAE"] if forecast_report else None,
    )

    logger.info(f"\n📊 RIESGO ACTUAL:")
    logger.info(f"   {current_risk['emoji']} Score: {current_risk['total_score']}/100 → {current_risk['level']}")
    for factor_name, factor_data in current_risk["factors"].items():
        logger.info(f"   • {factor_name}: {factor_data['raw_score']:.0f}/100 "
                     f"(peso: {factor_data['weight']:.0%})")

    results["current_risk"] = current_risk

    # =========================================================================
    # PASO 7: ALERTAS
    # =========================================================================
    print("\n" + "=" * 70)
    print("  🚨 PASO 7/8 — SISTEMA DE ALERTAS")
    print("=" * 70)

    alert_system = AlertSystem()
    alerts = alert_system.generate_alerts(
        current_risk,
        anomaly_summary=anomaly_summary,
    )

    # Mostrar reporte de alertas
    report = alert_system.format_alert_report(alerts)
    print("\n" + report)

    results["alerts"] = alerts
    results["alert_system"] = alert_system

    # =========================================================================
    # PASO 8: VISUALIZACIÓN
    # =========================================================================
    if not args.no_plots:
        print("\n" + "=" * 70)
        print("  📊 PASO 8/8 — GENERACIÓN DE GRÁFICOS")
        print("=" * 70)

        test_timestamps = df_ts.index[-len(y_test):]

        plot_paths = generate_all_plots(
            df=df_ts,
            anomaly_df=anomaly_df,
            risk_df=risk_df,
            y_true=y_true,
            y_pred=y_pred,
            timestamps=test_timestamps,
            training_history=training_history,
            show=args.show_plots,
        )

        results["plot_paths"] = plot_paths
    else:
        logger.info("⏭️ Gráficos omitidos (--no-plots)")

    # =========================================================================
    # REPORTE FINAL
    # =========================================================================
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("  ✅ PIPELINE COMPLETADO — REPORTE FINAL")
    print("=" * 70)

    print(f"""
    📊 AGRILION — Resumen del Análisis
    {'─' * 50}

    📦 Datos:
       • Registros procesados: {len(df_ts)}
       • Rango temporal: {df_ts.index[0].strftime('%d/%m/%Y')} → {df_ts.index[-1].strftime('%d/%m/%Y')}
       • Sensores: {', '.join(SENSOR_COLUMNS)}

    🧠 Modelo LSTM:
       • Arquitectura: {' → '.join(str(u) for u in LSTM_CONFIG['lstm_units'])} → {LSTM_CONFIG['dense_units']}
       • Secuencia de entrada: {LSTM_CONFIG['sequence_length']} pasos
       • R² promedio: {forecast_report['metrics']['global']['avg_R2']:.4f}
       • MAE promedio: {forecast_report['metrics']['global']['avg_MAE']:.4f}

    🔍 Anomalías:
       • Detectadas (consenso): {anomaly_df['is_anomaly'].sum()}/{len(anomaly_df)}
       • Tasa: {anomaly_df['is_anomaly'].mean() * 100:.1f}%

    ⚠️  Riesgo Actual:
       • Score: {current_risk['total_score']}/100
       • Nivel: {current_risk['emoji']} {current_risk['level']}

    🚨 Alertas: {len(alerts)} activas

    ⏱️  Tiempo total: {elapsed:.1f}s
    """)

    if not args.no_plots:
        print(f"    📁 Gráficos guardados en: {OUTPUTS_DIR}")
        for name, path in results.get("plot_paths", {}).items():
            print(f"       • {name}: {Path(path).name}")

    print(f"\n    💡 Para iniciar la API:")
    print(f"       cd src && uvicorn api.app:app --reload --port 8000")
    print(f"       Documentación: http://localhost:8000/docs\n")

    return results


if __name__ == "__main__":
    results = run_pipeline()
