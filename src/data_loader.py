"""
AGRILION — Data Loader
========================

Módulo de carga y limpieza de datos de sensores.
Soporta CSV y JSON, con validación de rangos físicos,
imputación de valores faltantes y conversión a series temporales.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, List
import logging

from .config import SENSOR_COLUMNS, SENSOR_RANGES, DEFAULT_CSV_PATH

logger = logging.getLogger(__name__)


def load_csv(path: Union[str, Path] = None) -> pd.DataFrame:
    """
    Carga datos de sensores desde un archivo CSV.

    Parameters
    ----------
    path : str or Path, optional
        Ruta al archivo CSV. Default: DEFAULT_CSV_PATH.

    Returns
    -------
    pd.DataFrame
        DataFrame con datos crudos, timestamp parseado como datetime.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    ValueError
        Si faltan columnas requeridas.
    """
    path = Path(path) if path else DEFAULT_CSV_PATH

    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    logger.info(f"Cargando datos desde: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])

    # Validar columnas mínimas
    required = {"timestamp"} | set(SENSOR_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en el CSV: {missing}")

    logger.info(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
    return df


def load_json(path: Union[str, Path]) -> pd.DataFrame:
    """
    Carga datos de sensores desde un archivo JSON.

    Parameters
    ----------
    path : str or Path
        Ruta al archivo JSON. Acepta formato records o columnar.

    Returns
    -------
    pd.DataFrame
        DataFrame con datos crudos.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    logger.info(f"Cargando datos JSON desde: {path}")
    df = pd.read_json(path)

    # Asegurar que timestamp sea datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def clean_data(
    df: pd.DataFrame,
    columns: List[str] = None,
    interpolation_method: str = "linear",
    max_consecutive_nan: int = 5,
    clip_to_physical: bool = True,
    remove_extreme_outliers: bool = True,
    outlier_sigma: float = 4.0,
) -> pd.DataFrame:
    """
    Limpia y preprocesa los datos de sensores.

    Pipeline de limpieza:
    1. Elimina filas completamente vacías
    2. Valida rangos físicos (clip o NaN)
    3. Detecta y reemplaza outliers extremos (>4σ)
    4. Imputa valores faltantes (interpolación lineal)
    5. Llena bordes con forward/backward fill

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos crudos.
    columns : list of str, optional
        Columnas de sensores a limpiar. Default: SENSOR_COLUMNS.
    interpolation_method : str
        Método de interpolación para imputar NaN. Default: 'linear'.
    max_consecutive_nan : int
        Máximo de NaN consecutivos a interpolar. Gaps más largos permanecen.
    clip_to_physical : bool
        Si True, recorta valores a rangos físicos válidos.
    remove_extreme_outliers : bool
        Si True, reemplaza outliers extremos (>outlier_sigma σ) con NaN.
    outlier_sigma : float
        Número de desviaciones estándar para considerar un outlier extremo.

    Returns
    -------
    pd.DataFrame
        DataFrame limpio con datos imputados.
    """
    columns = columns or SENSOR_COLUMNS
    df = df.copy()
    stats = {"original_rows": len(df)}

    # 1. Eliminar filas completamente vacías en columnas de sensores
    mask_all_nan = df[columns].isna().all(axis=1)
    df = df[~mask_all_nan].reset_index(drop=True)
    stats["dropped_empty"] = mask_all_nan.sum()

    # 2. Validar rangos físicos
    if clip_to_physical:
        for col in columns:
            if col in SENSOR_RANGES:
                valid_min = SENSOR_RANGES[col]["min"]
                valid_max = SENSOR_RANGES[col]["max"]

                # Marcar fuera de rango como NaN
                out_of_range = (df[col] < valid_min) | (df[col] > valid_max)
                n_out = out_of_range.sum()
                if n_out > 0:
                    logger.warning(
                        f"{col}: {n_out} valores fuera de rango "
                        f"[{valid_min}, {valid_max}] → NaN"
                    )
                    df.loc[out_of_range, col] = np.nan

    # 3. Detectar y eliminar outliers extremos
    if remove_extreme_outliers:
        for col in columns:
            valid_data = df[col].dropna()
            if len(valid_data) > 10:
                mean = valid_data.mean()
                std = valid_data.std()
                if std > 0:
                    z_scores = np.abs((df[col] - mean) / std)
                    outliers = z_scores > outlier_sigma
                    n_outliers = outliers.sum()
                    if n_outliers > 0:
                        logger.info(
                            f"{col}: {n_outliers} outliers extremos "
                            f"(>{outlier_sigma}σ) → NaN"
                        )
                        df.loc[outliers, col] = np.nan

    # 4. Contar NaN antes de imputar
    nan_before = df[columns].isna().sum().to_dict()
    stats["nan_before_imputation"] = nan_before

    # 5. Interpolación
    for col in columns:
        df[col] = df[col].interpolate(
            method=interpolation_method,
            limit=max_consecutive_nan,
            limit_direction="both",
        )

    # 6. Forward / backward fill para bordes
    for col in columns:
        df[col] = df[col].ffill().bfill()

    # Estadísticas finales
    nan_after = df[columns].isna().sum().to_dict()
    stats["nan_after_imputation"] = nan_after
    stats["final_rows"] = len(df)

    logger.info(
        f"Limpieza completada: {stats['original_rows']} → {stats['final_rows']} filas. "
        f"NaN imputados: {sum(nan_before.values()) - sum(nan_after.values())}"
    )

    return df


def to_timeseries(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    freq: str = None,
    resample_method: str = "mean",
) -> pd.DataFrame:
    """
    Convierte el DataFrame a formato de serie temporal con índice datetime.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columna timestamp.
    timestamp_col : str
        Nombre de la columna de timestamp.
    freq : str, optional
        Frecuencia objetivo (ej: '1h', '2h'). Si None, se infiere.
    resample_method : str
        Método de resampleo si se especifica freq. Default: 'mean'.

    Returns
    -------
    pd.DataFrame
        DataFrame con DatetimeIndex, frecuencia regular.
    """
    df = df.copy()

    # Asegurar datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Ordenar por tiempo
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    # Establecer como índice
    df = df.set_index(timestamp_col)

    # Inferir frecuencia si no se especifica
    if freq is None:
        inferred = pd.infer_freq(df.index[:100])
        if inferred:
            freq = inferred
            logger.info(f"Frecuencia inferida: {freq}")
        else:
            # Calcular mediana de diferencias
            diffs = df.index.to_series().diff().dropna()
            median_diff = diffs.median()
            logger.info(f"Frecuencia no inferida. Mediana de intervalo: {median_diff}")

    # Resamplear si se especifica frecuencia
    if freq:
        sensor_cols = [c for c in SENSOR_COLUMNS if c in df.columns]
        other_cols = [c for c in df.columns if c not in sensor_cols]

        if resample_method == "mean":
            df_resampled = df[sensor_cols].resample(freq).mean()
        elif resample_method == "median":
            df_resampled = df[sensor_cols].resample(freq).median()
        else:
            df_resampled = df[sensor_cols].resample(freq).first()

        # Mantener otras columnas (tomar el primer valor)
        for col in other_cols:
            df_resampled[col] = df[col].resample(freq).first()

        # Interpolar huecos del resampleo
        df_resampled[sensor_cols] = df_resampled[sensor_cols].interpolate(method="linear")
        df = df_resampled.dropna(subset=sensor_cols)

    logger.info(
        f"Serie temporal: {len(df)} puntos, "
        f"rango: {df.index.min()} → {df.index.max()}"
    )

    return df


def load_and_prepare(
    path: Union[str, Path] = None,
    file_format: str = "csv",
    freq: str = None,
) -> pd.DataFrame:
    """
    Pipeline completo: cargar → limpiar → convertir a serie temporal.

    Función de conveniencia que encadena load_csv/load_json → clean_data → to_timeseries.

    Parameters
    ----------
    path : str or Path, optional
        Ruta al archivo de datos.
    file_format : str
        Formato del archivo: 'csv' o 'json'.
    freq : str, optional
        Frecuencia para resampleo.

    Returns
    -------
    pd.DataFrame
        DataFrame limpio con DatetimeIndex.
    """
    # Cargar
    if file_format == "json":
        df = load_json(path)
    else:
        df = load_csv(path)

    # Limpiar
    df = clean_data(df)

    # Convertir a serie temporal
    df = to_timeseries(df, freq=freq)

    return df
