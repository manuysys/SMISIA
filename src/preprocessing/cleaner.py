"""
SMISIA — Limpieza e Imputación de Datos
Pipeline de preprocesamiento completo.
"""
import logging
import pandas as pd
import numpy as np
from src.config import get_config
from src.preprocessing.validators import (
    validate_schema,
    validate_timestamps,
    filter_physical_ranges,
)

logger = logging.getLogger("smisia.cleaner")


def impute_gaps(
    df: pd.DataFrame,
    linear_max_hours: int = 6,
    ffill_max_hours: int = 48,
) -> pd.DataFrame:
    """
    Imputación por silo_id:
    - <= linear_max_hours: interpolación lineal
    - <= ffill_max_hours: forward-fill → back-fill
    - > ffill_max_hours: dejar NaN (se usa como 'missing_pct')
    """
    sensor_cols = [
        "temperature_c", "humidity_pct", "co2_ppm",
        "nh3_ppm", "battery_pct",
    ]
    sensor_cols = [c for c in sensor_cols if c in df.columns]

    df = df.copy()
    df["imputed"] = False

    for silo_id, group in df.groupby("silo_id"):
        group = group.sort_values("timestamp")
        idx = group.index

        for col in sensor_cols:
            series = group[col].copy()
            original_nans = series.isna()

            if not original_nans.any():
                continue

            # Calcular tamaño de gaps
            is_nan = series.isna().values
            gap_sizes = _compute_gap_sizes(is_nan, group["timestamp"].values)

            # Fase 1: Interpolación lineal para gaps <= linear_max_hours
            small_gap_mask = (gap_sizes > 0) & (gap_sizes <= linear_max_hours)
            if small_gap_mask.any():
                temp_series = series.copy()
                temp_series[~small_gap_mask & original_nans] = np.nan
                interpolated = temp_series.interpolate(method="linear")
                series[small_gap_mask] = interpolated[small_gap_mask]

            # Fase 2: ffill/bfill para gaps <= ffill_max_hours
            medium_gap_mask = (
                (gap_sizes > linear_max_hours)
                & (gap_sizes <= ffill_max_hours)
                & series.isna()
            )
            if medium_gap_mask.any():
                filled = series.ffill().bfill()
                series[medium_gap_mask] = filled[medium_gap_mask]

            # Marcar imputados
            newly_filled = original_nans & series.notna()
            df.loc[idx[newly_filled], "imputed"] = True
            df.loc[idx, col] = series.values

    return df


def _compute_gap_sizes(is_nan: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Calcula el tamaño en horas de cada gap de NaN."""
    n = len(is_nan)
    gap_hours = np.zeros(n)

    if not is_nan.any():
        return gap_hours

    # Convertir timestamps a horas
    ts = pd.to_datetime(timestamps)

    i = 0
    while i < n:
        if is_nan[i]:
            # Encontrar inicio y fin del gap
            gap_start = i
            while i < n and is_nan[i]:
                i += 1
            gap_end = i  # primer no-NaN después del gap

            # Calcular duración del gap
            if gap_start > 0 and gap_end < n:
                hours = (ts[gap_end] - ts[gap_start - 1]).total_seconds() / 3600
            elif gap_start > 0:
                hours = (ts[-1] - ts[gap_start - 1]).total_seconds() / 3600
            else:
                hours = (ts[gap_end] - ts[0]).total_seconds() / 3600 if gap_end < n else 999

            # Asignar a cada posición del gap
            gap_hours[gap_start:gap_end] = hours
        else:
            i += 1

    return gap_hours


def run_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de preprocesamiento:
    1. Validar esquema
    2. Validar/normalizar timestamps
    3. Filtrar rangos físicos
    4. Imputar gaps
    """
    prep_cfg = get_config("preprocessing")

    logger.info(f"Inicio preprocesamiento: {len(df):,} filas")

    # 1. Schema
    df = validate_schema(df)

    # 2. Timestamps
    df = validate_timestamps(df)
    logger.info(f"Tras validar timestamps: {len(df):,} filas")

    # 3. Rangos físicos
    df = filter_physical_ranges(df)

    # 4. Imputación
    df = impute_gaps(
        df,
        linear_max_hours=prep_cfg["imputation"]["linear_interpolation_max_gap_hours"],
        ffill_max_hours=prep_cfg["imputation"]["ffill_bfill_max_gap_hours"],
    )

    # 5. Ordenar
    df = df.sort_values(["silo_id", "timestamp"]).reset_index(drop=True)

    logger.info(f"Preprocesamiento completo: {len(df):,} filas")
    return df
