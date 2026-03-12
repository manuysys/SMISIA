"""
SMISIA — Monitoreo de Deriva (Drift)
Implementa cálculo de PSI (Population Stability Index) y logging de métricas.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("smisia.monitoring")

def compute_psi(expected, actual, buckets=10):
    """
    Calcula el Population Stability Index (PSI) entre dos distribuciones.
    PSI = sum((Actual % - Expected %) * ln(Actual % / Expected %))
    Ranges:
    - < 0.1: No change
    - 0.1 - 0.25: Slight change
    - > 0.25: Significant change (Retrain recommended)
    """
    def scale_range(input_data, min_val, max_val):
        return (input_data - min_val) / (max_val - min_val + 1e-7)

    def sub_compute_psi(e_perc, a_perc):
        if a_perc == 0:
            a_perc = 0.0001
        if e_perc == 0:
            e_perc = 0.0001
        return (a_perc - e_perc) * np.log(a_perc / e_perc)

    # Bucketing
    min_val = min(np.min(expected), np.min(actual))
    max_val = max(np.max(expected), np.max(actual))
    
    e_scaled = scale_range(expected, min_val, max_val)
    a_scaled = scale_range(actual, min_val, max_val)
    
    e_counts = np.histogram(e_scaled, bins=buckets, range=(0, 1))[0]
    a_counts = np.histogram(a_scaled, bins=buckets, range=(0, 1))[0]
    
    e_perc = e_counts / len(expected)
    a_perc = a_counts / len(actual)
    
    psi_val = np.sum([sub_compute_psi(e_perc[i], a_perc[i]) for i in range(len(e_perc))])
    return psi_val

def check_feature_drift(df_baseline, df_actual, feature_cols, threshold=0.25):
    """
    Verifica drift en múltiples features.
    """
    results = {}
    for col in feature_cols:
        if col in df_baseline.columns and col in df_actual.columns:
            psi = compute_psi(df_baseline[col].values, df_actual[col].values)
            results[col] = {
                "psi": round(psi, 4),
                "alert": psi > threshold
            }
    return results

def log_silo_metrics(df, silo_id):
    """
    Calcula y loguea métricas específicas por silo.
    """
    silo_data = df[df["silo_id"] == silo_id]
    if silo_data.empty:
        return None
        
    metrics = {
        "silo_id": silo_id,
        "n_readings": len(silo_data),
        "prediction_dist": silo_data["label"].value_counts(normalize=True).to_dict(),
        "missing_rate": silo_data[["temperature_c", "humidity_pct", "co2_ppm"]].isna().mean().mean(),
        "avg_sensor_health": silo_data["sensor_health"].mean() if "sensor_health" in silo_data.columns else None
    }
    
    logger.info(f"Métricas Silo {silo_id}: {metrics}")
    return metrics
