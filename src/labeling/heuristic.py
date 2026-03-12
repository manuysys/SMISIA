"""
SMISIA — Labeling Heurístico
Genera etiquetas proxy cuando no hay labels reales.
"""

import logging
import pandas as pd
import numpy as np
from src.config import get_config

logger = logging.getLogger("smisia.labeling")


def apply_heuristic_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica reglas heurísticas para generar etiquetas de estado:
    - CRÍTICO: humidity > 25% AND co2 > 2000 AND humidity_delta_24h > 3pp;
               OR temp > 45°C AND co2 elevado
    - PROBLEMA: humidity 18-25% con tendencia creciente OR co2 1200-2000 sostenido
    - TOLERABLE: picos breves fuera de rango < 48h
    - BIEN: estable dentro de rangos saludables
    """
    lab_cfg = get_config("labeling")
    df = df.copy()

    # Inicializar como "bien"
    df["heuristic_label"] = "bien"

    # ------ CRÍTICO ------
    crit = lab_cfg["critico"]

    # Condición 1: humidity + CO2 + delta
    hum_col_24h_slope = "humidity_pct_24h_slope"
    cond_crit_1 = (df["humidity_pct"] > crit["humidity_min"]) & (
        df["co2_ppm"] > crit["co2_min"]
    )
    if hum_col_24h_slope in df.columns:
        # delta_24h > 3 pp → slope * 24 > 3
        cond_crit_1 = cond_crit_1 & (
            df[hum_col_24h_slope] * 24 > crit["humidity_delta_24h_min"]
        )

    # Condición 2: temp extrema + CO2
    cond_crit_2 = (df["temperature_c"] > crit["temperature_max_alt"]) & (
        df["co2_ppm"] > crit["co2_elevated_alt"]
    )

    df.loc[cond_crit_1 | cond_crit_2, "heuristic_label"] = "critico"

    # ------ PROBLEMA ------
    prob = lab_cfg["problema"]
    hum_range = prob["humidity_range"]
    co2_range = prob["co2_range"]

    cond_prob_hum = (df["humidity_pct"] >= hum_range[0]) & (
        df["humidity_pct"] <= hum_range[1]
    )
    if hum_col_24h_slope in df.columns:
        cond_prob_hum = cond_prob_hum & (df[hum_col_24h_slope] > 0)

    cond_prob_co2 = (df["co2_ppm"] >= co2_range[0]) & (df["co2_ppm"] <= co2_range[1])

    # Solo aplicar si no es ya crítico
    is_not_critical = df["heuristic_label"] != "critico"
    df.loc[(cond_prob_hum | cond_prob_co2) & is_not_critical, "heuristic_label"] = (
        "problema"
    )

    # ------ TOLERABLE ------
    # Lecturas ligeramente fuera de rango
    cond_tol = (
        ((df["humidity_pct"] > 14) & (df["humidity_pct"] <= hum_range[0]))
        | ((df["co2_ppm"] > 800) & (df["co2_ppm"] <= co2_range[0]))
        | ((df["temperature_c"] > 35) & (df["temperature_c"] <= 45))
    )
    is_bien = df["heuristic_label"] == "bien"
    df.loc[cond_tol & is_bien, "heuristic_label"] = "tolerable"

    # Marcar fuente
    df["label_source"] = "heuristic_v1"

    # Estadísticas
    counts = df["heuristic_label"].value_counts()
    for label, count in counts.items():
        pct = 100.0 * count / len(df)
        logger.info(f"Label '{label}': {count:,} ({pct:.1f}%)")

    return df


def select_for_active_learning(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    top_n: int = 100,
) -> pd.DataFrame:
    """
    Selecciona los top-N casos con menor confianza del modelo
    para revisión humana (active learning).

    Args:
        df: DataFrame con datos
        probabilities: array (n_samples, n_classes) de probabilidades
        top_n: cantidad de muestras a seleccionar
    Returns:
        DataFrame con las muestras más inciertas
    """
    # Confianza = max probability
    confidence = probabilities.max(axis=1)

    # Tomar los que tienen menor confianza
    indices = np.argsort(confidence)[:top_n]
    selected = df.iloc[indices].copy()
    selected["model_confidence"] = confidence[indices]
    selected["needs_review"] = True

    logger.info(
        f"Active learning: seleccionados {len(selected)} casos "
        f"(confianza min: {confidence[indices].min():.3f}, "
        f"max: {confidence[indices].max():.3f})"
    )

    return selected


def encode_labels(labels: pd.Series) -> tuple:
    """
    Codifica labels a enteros.
    Returns: (encoded, mapping)
    """
    mapping = {"bien": 0, "tolerable": 1, "problema": 2, "critico": 3}
    encoded = labels.map(mapping)
    return encoded, mapping


def decode_labels(encoded: np.ndarray, mapping: dict = None) -> np.ndarray:
    """Decodifica enteros a labels de texto."""
    if mapping is None:
        mapping = {"bien": 0, "tolerable": 1, "problema": 2, "critico": 3}
    reverse = {v: k for k, v in mapping.items()}
    return np.array([reverse.get(int(v), "desconocido") for v in encoded])
