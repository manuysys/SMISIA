"""
SMISIA — Active Learning Selection
Identifica muestras con alta incertidumbre o entropía para revisión humana.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("smisia.active_learning")

def select_uncertain_samples(df, probabilities, uncertainty_std, n_samples=50):
    """
    Selecciona muestras basadas en incertidumbre (std del ensemble) y entropía.
    Muestras con mayor std son candidatas a revisión humana.
    """
    df = df.copy()
    
    # 1. Por Entropía (probalidades planas)
    # Entropy = -sum(p * log(p))
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
    
    df["prediction_entropy"] = entropy
    df["prediction_uncertainty_std"] = uncertainty_std
    
    # Marcamos para revisión si:
    # - Incertidumbre std > 0.15 O
    # - Entropía > 0.8 (clases muy similares)
    df["needs_human_review"] = (df["prediction_uncertainty_std"] > 0.15) | (df["prediction_entropy"] > 0.8)
    
    # Seleccionamos las N muestras más inciertas
    top_uncertain = df.sort_values("prediction_uncertainty_std", ascending=False).head(n_samples)
    
    logger.info(f"Active Learning: {df['needs_human_review'].sum()} muestras marcadas para revisión.")
    return top_uncertain

def export_for_labeling(df_uncertain, output_path="data/active_learning_batch.csv"):
    """
    Exporta muestras a CSV para que un experto las etiquete.
    """
    cols_to_export = [
        "silo_id", "timestamp", "temperature_c", "humidity_pct", 
        "co2_ppm", "predicted_label", "prediction_uncertainty_std"
    ]
    df_uncertain[cols_to_export].to_csv(output_path, index=False)
    logger.info(f"Lote de Active Learning exportado a {output_path}")
