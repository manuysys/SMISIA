"""
SMISIA — Feature Engineering
Calcula features agregadas en ventanas móviles para cada silo.
"""

import logging
import pandas as pd
import numpy as np
from src.config import get_config

logger = logging.getLogger("smisia.features")


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada silo_id y timestamp, calcula features en ventanas móviles.
    Ventanas: 6h, 24h, 72h, 7d (168h).
    """
    feat_cfg = get_config("features")
    windows = feat_cfg["windows_hours"]
    primary_vars = feat_cfg["primary_variables"]
    primary_vars = [v for v in primary_vars if v in df.columns]

    df = df.sort_values(["silo_id", "timestamp"]).copy()
    df = df.set_index("timestamp")

    all_features = []

    for silo_id, group in df.groupby("silo_id"):
        group = group.sort_index()
        # Usamos un diccionario para recolectar columnas y evitar fragmentación
        new_cols = {}

        for window_h in windows:
            window_str = f"{window_h}h"
            win = f"{window_h}h"

            for var in primary_vars:
                if var not in group.columns:
                    continue

                series = group[var]
                prefix = f"{var}_{window_str}"

                if series.isna().all():
                    # Si toda la serie es NaN, los resultados rolling también lo son
                    nan_series = pd.Series(np.nan, index=group.index)
                    new_cols[f"{prefix}_mean"] = nan_series
                    new_cols[f"{prefix}_median"] = nan_series
                    new_cols[f"{prefix}_min"] = nan_series
                    new_cols[f"{prefix}_max"] = nan_series
                    new_cols[f"{prefix}_std"] = pd.Series(0.0, index=group.index)
                    new_cols[f"{prefix}_skew"] = pd.Series(0.0, index=group.index)
                    new_cols[f"{prefix}_slope"] = nan_series
                    new_cols[f"{prefix}_count_missing"] = pd.Series(
                        float(window_h), index=group.index
                    )  # Aproximación
                    new_cols[f"{prefix}_pct_missing"] = pd.Series(
                        1.0, index=group.index
                    )
                    continue

                # Estadísticas básicas
                rolling = series.rolling(win, min_periods=1)
                new_cols[f"{prefix}_mean"] = rolling.mean()
                new_cols[f"{prefix}_median"] = rolling.median()
                new_cols[f"{prefix}_min"] = rolling.min()
                new_cols[f"{prefix}_max"] = rolling.max()
                new_cols[f"{prefix}_std"] = rolling.std().fillna(0)
                new_cols[f"{prefix}_skew"] = rolling.skew().fillna(0)

                # Slope (valor al final - valor al inicio) / ventana en horas
                first_val = series.rolling(win, min_periods=1).apply(
                    lambda x: x.iloc[0] if len(x) > 0 else np.nan, raw=False
                )
                new_cols[f"{prefix}_slope"] = (series - first_val) / max(
                    window_h, 1
                )

                # Missing count y porcentaje
                nan_count = series.isna().rolling(win, min_periods=1).sum()
                rolling_size = series.rolling(win, min_periods=1).apply(
                    len, raw=False
                )
                new_cols[f"{prefix}_count_missing"] = nan_count
                new_cols[f"{prefix}_pct_missing"] = nan_count / rolling_size.clip(
                    lower=1
                )

        silo_feats = pd.DataFrame(new_cols, index=group.index)
        silo_feats["silo_id"] = silo_id
        all_features.append(silo_feats)

    result = pd.concat(all_features)
    result = result.reset_index()

    # Merge con DataFrame original
    df = df.reset_index()
    df = df.merge(result, on=["silo_id", "timestamp"], how="left")

    return df


def compute_humidity_counters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula contadores de humedad:
    - hours_humidity_above_X_24h para X en [16, 18, 20]
    - consecutive_hours_humidity_increase_24h
    """
    feat_cfg = get_config("features")
    thresholds = feat_cfg["humidity_thresholds_pct"]

    df = df.sort_values(["silo_id", "timestamp"]).copy()
    all_counters = []

    for silo_id, group in df.groupby("silo_id"):
        idx = group.index
        humidity = group["humidity_pct"].values
        new_cols = {}

        for thresh in thresholds:
            col_name = f"hours_humidity_above_{thresh}_24h"
            above = (humidity > thresh).astype(float)
            # Ventana de 24h (12 lecturas si cada 2h)
            window_size = min(12, len(above))
            counts = pd.Series(above).rolling(window_size, min_periods=1).sum()
            new_cols[col_name] = counts.values * 2  # * 2h per reading

        # Consecutive hours of humidity increase
        col_name = "consecutive_hours_humidity_increase_24h"
        diffs = np.diff(humidity, prepend=humidity[0])
        increasing = (diffs > 0).astype(int)
        # Max run in 24h window
        consec = (
            pd.Series(increasing)
            .rolling(12, min_periods=1)
            .apply(_max_consecutive_ones, raw=True)
        )
        new_cols[col_name] = consec.values * 2

        silo_counters = pd.DataFrame(new_cols, index=idx)
        silo_counters["silo_id"] = silo_id
        silo_counters["timestamp"] = group["timestamp"]
        all_counters.append(silo_counters)

    counters_df = pd.concat(all_counters)
    df = df.merge(counters_df, on=["silo_id", "timestamp"], how="left")

    return df


def _max_consecutive_ones(arr: np.ndarray) -> float:
    """Calcula la racha máxima de 1s en un array."""
    max_run = 0
    current_run = 0
    for val in arr:
        if val == 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return float(max_run)


def compute_combined_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Señales combinadas booleanas:
    - temp_and_humidity_up_24h
    - co2_spike_recent
    """
    feat_cfg = get_config("features")
    slope_thresholds = feat_cfg["slope_thresholds"]
    co2_spike_sigma = feat_cfg["co2_spike_sigma"]

    df = df.copy()

    # temp_and_humidity_up_24h
    temp_slope_col = "temperature_c_24h_slope"
    hum_slope_col = "humidity_pct_24h_slope"

    if temp_slope_col in df.columns and hum_slope_col in df.columns:
        df["temp_and_humidity_up_24h"] = (
            (df[temp_slope_col] > slope_thresholds.get("temperature_c", 0.5))
            & (df[hum_slope_col] > slope_thresholds.get("humidity_pct", 0.3))
        ).astype(int)
    else:
        df["temp_and_humidity_up_24h"] = 0

    # co2_spike_recent (max CO2 6h > baseline + 3*std)
    co2_max_6h = "co2_ppm_6h_max"
    co2_mean_168h = "co2_ppm_168h_mean"
    co2_std_168h = "co2_ppm_168h_std"

    if all(c in df.columns for c in [co2_max_6h, co2_mean_168h, co2_std_168h]):
        df["co2_spike_recent"] = (
            df[co2_max_6h] > df[co2_mean_168h] + co2_spike_sigma * df[co2_std_168h]
        ).astype(int)
    else:
        df["co2_spike_recent"] = 0

    return df


def compute_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features estáticas y de salud:
    - days_since_fill
    - rssi_mean_24h, snr_mean_24h
    - pct_imputed_72h
    """
    df = df.sort_values(["silo_id", "timestamp"]).copy()

    # days_since_fill
    if "fill_date" in df.columns:
        fill_dates = pd.to_datetime(df["fill_date"], errors="coerce", utc=True, format="ISO8601")
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        df["days_since_fill"] = (
            df["timestamp"] - fill_dates
        ).dt.total_seconds() / 86400
        df["days_since_fill"] = df["days_since_fill"].clip(lower=0).fillna(0)
    else:
        df["days_since_fill"] = 0

    # RSSI y SNR medias 24h
    all_static = []
    for silo_id, group in df.groupby("silo_id"):
        idx = group.index
        new_cols = {}
        if "rssi" in group.columns:
            new_cols["rssi_mean_24h"] = (
                group["rssi"].rolling(12, min_periods=1).mean().values
            )
        if "snr" in group.columns:
            new_cols["snr_mean_24h"] = (
                group["snr"].rolling(12, min_periods=1).mean().values
            )
        if "imputed" in group.columns:
            imp_series = group["imputed"].astype(float)
            new_cols["pct_imputed_72h"] = (
                imp_series.rolling(36, min_periods=1).mean().values
            )
        
        if new_cols:
            silo_static = pd.DataFrame(new_cols, index=idx)
            silo_static["silo_id"] = silo_id
            silo_static["timestamp"] = group["timestamp"]
            all_static.append(silo_static)

    if all_static:
        static_df = pd.concat(all_static)
        df = df.merge(static_df, on=["silo_id", "timestamp"], how="left")

    return df


def compute_sensor_health(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula una métrica de salud del sensor (0-1):
    - Basada en RSSI, SNR y Nivel de Batería.
    """
    df = df.copy()
    
    # Normalizar componentes (0..1)
    # RSSI: -120 (0) a -60 (1)
    rssi_norm = ((df["rssi"].clip(-120, -60) + 120) / 60).fillna(0.5)
    # SNR: 0 (0) a 15 (1)
    snr_norm = (df["snr"].clip(0, 15) / 15).fillna(0.5)
    # Battery: 0..100 -> 0..1
    batt_norm = (df["battery_pct"].clip(0, 100) / 100).fillna(0.5)
    
    # Salud = promedio ponderado
    df["sensor_health"] = (rssi_norm * 0.3 + snr_norm * 0.3 + batt_norm * 0.4)
    return df


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de feature engineering."""
    logger.info("Inicio feature engineering...")

    # Activar opción para evitar FutureWarning en fillna
    pd.set_option('future.no_silent_downcasting', True)

    df = compute_rolling_features(df)
    logger.info("Rolling features calculadas")

    df = compute_humidity_counters(df)
    logger.info("Contadores de humedad calculados")

    df = compute_combined_signals(df)
    logger.info("Señales combinadas calculadas")

    df = compute_static_features(df)
    logger.info("Features estáticas calculadas")
    
    df = compute_sensor_health(df)
    logger.info("Salud del sensor calculada")

    # Rellenar NaN restantes con 0 para features calculadas
    feature_prefixes = [
        "temperature_c_",
        "humidity_pct_",
        "co2_ppm_",
        "nh3_ppm_",
        "battery_pct_",
        "hours_humidity",
        "consecutive_hours",
        "temp_and_humidity_up",
        "co2_spike_recent",
        "days_since_fill",
        "rssi_mean",
        "snr_mean",
        "pct_imputed",
        "sensor_health",
        "imputed", # Lo incluimos como feature binaria
    ]
    
    feature_cols = [
        c
        for c in df.columns
        if any(c.startswith(p) for p in feature_prefixes) or c == "imputed"
    ]
    df[feature_cols] = df[feature_cols].fillna(0).infer_objects(copy=False)

    logger.info(f"Feature engineering completo: {len(feature_cols)} features generadas")
    return df
