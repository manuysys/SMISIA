"""
SMISIA — Validador de Calidad de Datos
Comprobaciones previas a la inferencia para asegurar integridad.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("smisia.validator")

def validate_inference_data(df: pd.DataFrame, feature_columns: list) -> dict:
    """
    Valida que los datos de entrada para inferencia sean correctos.
    Retorna un dict con {'valid': bool, 'errors': list, 'warnings': list}
    """
    errors = []
    warnings = []
    
    # 1. Verificar columnas faltantes
    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        errors.append(f"Faltan columnas requeridas: {missing_cols}")
    
    # 2. Verificar tipos de datos
    for col in feature_columns:
        if col in df.columns:
            if not np.issubdtype(df[col].dtype, np.number):
                errors.append(f"Columna {col} no es numérica: {df[col].dtype}")
    
    # 3. Verificar valores fuera de rango físicos (ejemplo)
    ranges = {
        "temperature_c": (-20, 80),
        "humidity_pct": (0, 100),
        "co2_ppm": (0, 50000),
        "nh3_ppm": (0, 1000)
    }
    
    for col, (min_v, max_v) in ranges.items():
        if col in df.columns:
            out_of_range = df[(df[col] < min_v) | (df[col] > max_v)]
            if not out_of_range.empty:
                warnings.append(f"Valores de {col} fuera de rango físico: {len(out_of_range)} muestras")

    # 4. Verificar porcentaje de nulos
    null_pct = df[feature_columns].isnull().mean().max()
    if null_pct > 0.5:
        errors.append(f"Demasiados valores nulos: {null_pct:.1%}")
    elif null_pct > 0.1:
        warnings.append(f"Porcentaje de nulos significativo: {null_pct:.1%}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

def clean_data_for_inference(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Aplica limpiezas básicas extremas antes de pasar al modelo."""
    df_clean = df.copy()
    # Rellenar nulos con 0 o medias móviles si existieran
    df_clean[feature_columns] = df_clean[feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
    return df_clean
