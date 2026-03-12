"""
SMISIA — Validadores de Datos
Validación de esquema, rangos físicos y timestamps.
"""

import logging
import pandas as pd
import numpy as np
from src.config import get_config

logger = logging.getLogger("smisia.validators")


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Verifica que el DataFrame tenga las columnas requeridas."""
    required = [
        "silo_id",
        "timestamp",
        "temperature_c",
        "humidity_pct",
        "co2_ppm",
        "battery_pct",
        "rssi",
        "snr",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes: {missing}")

    # Columnas opcionales
    optional = ["nh3_ppm", "fill_date", "label", "label_source"]
    for col in optional:
        if col not in df.columns:
            logger.info(f"Columna opcional ausente: {col}")

    return df


def validate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida y normaliza timestamps a UTC.
    Rechaza filas con timestamps inválidos.
    """
    original_len = len(df)
    df = df.copy()

    # Parsear timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True, format="ISO8601")

    # Rechazar filas con timestamp inválido
    invalid_mask = df["timestamp"].isna()
    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        logger.warning(
            f"Rechazadas {n_invalid}/{original_len} filas con timestamp inválido"
        )
        df = df[~invalid_mask].copy()

    return df


def filter_physical_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina lecturas fuera de rango físico.
    Registra en logs las filas eliminadas.
    """
    ranges = get_config("physical_ranges")
    df = df.copy()
    total_removed = 0

    for col, limits in ranges.items():
        if col not in df.columns:
            continue
        col_min = limits["min"]
        col_max = limits["max"]
        mask = df[col].notna() & ((df[col] < col_min) | (df[col] > col_max))
        n_out = mask.sum()
        if n_out > 0:
            logger.warning(
                f"Lecturas fuera de rango en {col}: {n_out} "
                f"(rango válido: [{col_min}, {col_max}])"
            )
            df.loc[mask, col] = np.nan
            total_removed += n_out

    if total_removed > 0:
        logger.info(f"Total lecturas fuera de rango convertidas a NaN: {total_removed}")

    return df


def check_data_sufficiency(
    df: pd.DataFrame,
    silo_id: str,
    window_days: int = 7,
    threshold: float = 0.30,
) -> dict:
    """
    Verifica si hay datos suficientes en la ventana.
    Retorna dict con flags de suficiencia por variable.
    """
    key_vars = ["temperature_c", "humidity_pct"]
    result = {"sufficient": True, "details": {}}

    silo_data = df[df["silo_id"] == silo_id].copy()
    if silo_data.empty:
        return {"sufficient": False, "details": {"error": "silo no encontrado"}}

    latest = silo_data["timestamp"].max()
    window_start = latest - pd.Timedelta(days=window_days)
    window_data = silo_data[silo_data["timestamp"] >= window_start]

    for var in key_vars:
        if var not in window_data.columns:
            result["details"][var] = {"missing_pct": 1.0, "sufficient": False}
            result["sufficient"] = False
            continue

        missing_pct = window_data[var].isna().mean()
        is_sufficient = missing_pct <= threshold
        result["details"][var] = {
            "missing_pct": round(float(missing_pct), 4),
            "sufficient": is_sufficient,
        }
        if not is_sufficient:
            result["sufficient"] = False
            logger.warning(
                f"Insuficientes datos para {silo_id}/{var}: "
                f"{missing_pct:.1%} missing en {window_days}d"
            )

    return result
