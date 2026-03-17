"""
SMISIA — Generador de Dataset Sintético Realista
Genera lecturas de sensores para 50 silobolsas durante 120 días.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Añadir raíz del proyecto al path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402


def generate_silo_readings(
    silo_id: str,
    start_date: pd.Timestamp,
    n_days: int,
    readings_per_day: int,
    rng: np.random.Generator,
    scenario: str = "normal",
) -> pd.DataFrame:
    """Genera lecturas para un silo con un escenario dado."""
    n_readings = n_days * readings_per_day
    freq_hours = 24 // readings_per_day
    timestamps = pd.date_range(
        start=start_date,
        periods=n_readings,
        freq=f"{freq_hours}h",
    )

    # Fill date: entre 1 y 30 días antes del start
    fill_offset = rng.integers(1, 30)
    fill_date = start_date - timedelta(days=int(fill_offset))

    # -- Baselines --
    base_temp = rng.uniform(18, 28)
    base_humidity = rng.uniform(10, 15)
    base_co2 = rng.uniform(300, 600)
    base_nh3 = rng.uniform(1, 10)
    base_battery = rng.uniform(85, 100)
    base_rssi = rng.integers(-120, -60)
    base_snr = rng.uniform(5, 15)

    # -- Time index normalizado 0..1 --
    t_norm = np.linspace(0, 1, n_readings)

    # -- Temperatura --
    temp_seasonal = 3 * np.sin(2 * np.pi * t_norm * (n_days / 365))
    temp_diurnal = 2 * np.sin(
        2 * np.pi * np.arange(n_readings) / readings_per_day
    )
    temp_noise = rng.normal(0, 0.5, n_readings)
    temperature = base_temp + temp_seasonal + temp_diurnal + temp_noise

    # -- Humedad --
    # Diurnal cycle (humedad sube cuando temp baja)
    hum_diurnal = -1 * np.sin(
        2 * np.pi * np.arange(n_readings) / readings_per_day
    )
    humidity_noise = rng.normal(0, 0.3, n_readings)
    humidity = base_humidity + hum_diurnal + humidity_noise

    # -- CO2 --
    # Correlacionado con temperatura y humedad (actividad biológica)
    co2_trend = (temperature - base_temp) * 10 + (humidity - base_humidity) * 5
    co2_noise = rng.normal(0, 20, n_readings)
    co2 = base_co2 + co2_trend + co2_noise

    # -- NH3 --
    nh3_noise = rng.normal(0, 0.5, n_readings)
    nh3 = base_nh3 + nh3_noise

    # -- Battery (descarga lenta) --
    battery_drain = np.linspace(0, rng.uniform(5, 20), n_readings)
    battery_noise = rng.normal(0, 0.2, n_readings)
    battery = base_battery - battery_drain + battery_noise

    # -- RSSI y SNR --
    rssi = base_rssi + rng.normal(0, 3, n_readings)
    snr = base_snr + rng.normal(0, 1, n_readings)

    # -----------------------------------------------------------------------
    # Aplicar escenarios de deterioro y deriva
    # -----------------------------------------------------------------------
    labels = np.full(n_readings, "bien", dtype=object)

    if scenario == "gradual_humidity":
        # Incremento gradual de humedad a partir de un punto aleatorio
        onset = rng.integers(n_readings // 4, n_readings // 2)
        ramp = np.zeros(n_readings)
        ramp[onset:] = np.linspace(0, rng.uniform(8, 18), n_readings - onset)
        humidity += ramp
        # CO2 sube fuertemente correlacionado (fermentación probable)
        co2[onset:] += ramp[onset:] * rng.uniform(50, 120)
        # Labels más realistas y progresivos (Lead Time)
        for i in range(onset, n_readings):
            h = humidity[i]
            c = co2[i]
            # Agregar ruido a la percepción del label para evitar AUC=1 perfecto
            h_noisy = h + rng.normal(0, 1.5)
            c_noisy = c + rng.normal(0, 200)
            
            if h_noisy > 28 and c_noisy > 2800:
                labels[i] = "critico"
            elif h_noisy > 22 or c_noisy > 1800:
                labels[i] = "problema"
            elif h_noisy > 18 or c_noisy > 1200:
                labels[i] = "tolerable"

    elif scenario == "sensor_drift":
        # Simula deriva del sensor de CO2 (falla técnica, no silo)
        # Sube sin que suban T o H
        onset = rng.integers(n_readings // 4, n_readings // 2)
        drift = np.linspace(0, 5000, n_readings - onset)
        co2[onset:] += drift
        # En este caso, el label NO debería ser crítico si T/H están bien
        # Pero el sistema de labels heurísticos podría confundirse (test de Robustez)
        for i in range(onset, n_readings):
            if co2[i] > 3000:
                labels[i] = "tolerable" # Alarma técnica, no necesariamente crítica de grano

    elif scenario == "temperature_spike":
        # Pico de temperatura (simula fermentación)
        spike_start = rng.integers(n_readings // 3, 2 * n_readings // 3)
        spike_len = rng.integers(24, 72)
        spike_end = min(spike_start + spike_len, n_readings)
        spike_mag = rng.uniform(10, 25)
        temperature[spike_start:spike_end] += spike_mag
        co2[spike_start:spike_end] += spike_mag * rng.uniform(30, 80)
        for i in range(spike_start, spike_end):
            t = temperature[i] + rng.normal(0, 2.0)
            c = co2[i] + rng.normal(0, 150)
            if t > 48 and c > 1800:
                labels[i] = "critico"
            elif t > 40 or c > 1400:
                labels[i] = "problema"
            else:
                labels[i] = "tolerable"

    elif scenario == "co2_rise":
        # Aumento sostenido de CO2
        onset = rng.integers(n_readings // 4, n_readings // 2)
        ramp = np.zeros(n_readings)
        ramp[onset:] = np.linspace(0, rng.uniform(1000, 2500), n_readings - onset)
        co2 += ramp
        humidity[onset:] += ramp[onset:] * 0.003  # leve correlación
        for i in range(onset, n_readings):
            c = co2[i] + rng.normal(0, 100)
            h = humidity[i] + rng.normal(0, 1.0)
            if c > 2200 and h > 26:
                labels[i] = "critico"
            elif c > 1400:
                labels[i] = "problema"
            elif c > 900:
                labels[i] = "tolerable"

    elif scenario == "sudden_anomaly":
        # Anomalías repentinas cortas
        n_anomalies = rng.integers(2, 6)
        for _ in range(n_anomalies):
            start = rng.integers(0, n_readings - 48)
            length = rng.integers(6, 48)
            end = min(start + length, n_readings)
            anom_type = rng.choice(["temp", "humidity", "co2"])
            if anom_type == "temp":
                temperature[start:end] += rng.uniform(10, 20)
                labels[start:end] = np.where(
                    temperature[start:end] > 45, "critico", "tolerable"
                )
            elif anom_type == "humidity":
                humidity[start:end] += rng.uniform(8, 20)
                labels[start:end] = np.where(
                    humidity[start:end] > 25, "problema", "tolerable"
                )
            else:
                co2[start:end] += rng.uniform(500, 2000)
                labels[start:end] = np.where(
                    co2[start:end] > 2000, "critico", "problema"
                )

    elif scenario == "sensor_noise":
        # Sensor ruidoso pero sin problema real
        temperature += rng.normal(0, 3, n_readings)
        humidity += rng.normal(0, 2, n_readings)
        co2 += rng.normal(0, 50, n_readings)
        # Algunos picos breves → tolerable
        for _ in range(rng.integers(1, 4)):
            s = rng.integers(0, n_readings - 12)
            labels[s: s + rng.integers(3, 12)] = "tolerable"

    # Clampear valores a rangos físicos
    temperature = np.clip(temperature, -40, 80)
    humidity = np.clip(humidity, 0, 100)
    co2 = np.clip(co2, 0, 50000)
    nh3 = np.clip(nh3, 0, 500)
    battery = np.clip(battery, 0, 100)

    df = pd.DataFrame(
        {
            "silo_id": silo_id,
            "timestamp": timestamps,
            "temperature_c": np.round(temperature, 2),
            "humidity_pct": np.round(humidity, 2),
            "co2_ppm": np.round(co2, 1),
            "nh3_ppm": np.round(nh3, 2),
            "battery_pct": np.round(battery, 1),
            "rssi": np.round(rssi).astype(int),
            "snr": np.round(snr, 2),
            "fill_date": fill_date.isoformat(),
            "label": labels,
        }
    )
    return df


def inject_missing_values(df: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Inyecta valores faltantes aleatorios."""
    sensor_cols = ["temperature_c", "humidity_pct", "co2_ppm", "nh3_ppm"]
    df = df.copy()
    for col in sensor_cols:
        mask = rng.random(len(df)) < rate
        df.loc[mask, col] = np.nan
    return df


def generate_dataset(config: dict = None) -> pd.DataFrame:
    """Genera el dataset completo sintético."""
    if config is None:
        config = load_config()

    syn_cfg = config["synthetic"]
    seed = config["project"]["random_seed"]
    rng = np.random.default_rng(seed)

    n_silos = syn_cfg["n_silos"]
    n_days = syn_cfg["days"]
    readings_per_day = syn_cfg["readings_per_day"]
    start_date = pd.Timestamp(syn_cfg["start_date"])
    missing_rate = syn_cfg["missing_rate"]

    # Distribuir escenarios para lograr la distribución de labels deseada
    # ~70% bien, 15% tolerable, 10% problema, 5% critico
    scenarios = (
        ["normal"] * 24          # 48% silos normales → mayormente "bien"
        + ["sensor_noise"] * 6   # 12% → algo de "tolerable"
        + ["gradual_humidity"] * 7   # 14% → tolerable/problema/critico
        + ["temperature_spike"] * 4  # 8% → tolerable/problema/critico
        + ["co2_rise"] * 4           # 8% → tolerable/problema
        + ["sudden_anomaly"] * 3     # 6% → mezcla
        + ["sensor_drift"] * 2       # 4% → Alarma técnica de CO2
    )
    rng.shuffle(scenarios)

    all_dfs = []
    for i in range(n_silos):
        silo_id = f"SILO_{i+1:03d}"
        scenario = scenarios[i] if i < len(scenarios) else "normal"
        # Variar start_date ligeramente
        offset_days = int(rng.integers(0, 5))
        silo_start = start_date + timedelta(days=offset_days)

        df = generate_silo_readings(
            silo_id=silo_id,
            start_date=silo_start,
            n_days=n_days,
            readings_per_day=readings_per_day,
            rng=rng,
            scenario=scenario,
        )
        all_dfs.append(df)

    dataset = pd.concat(all_dfs, ignore_index=True)

    # Inyectar missing values
    dataset = inject_missing_values(dataset, missing_rate, rng)

    # Agregar label_source
    dataset["label_source"] = "heuristic_v1"

    # Shuffle y ordenar por silo y timestamp
    dataset = dataset.sort_values(["silo_id", "timestamp"]).reset_index(drop=True)

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generar dataset sintético SMISIA")
    parser.add_argument(
        "--config", type=str, default=None, help="Ruta a config.yml"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta de salida (default: data/synthetic_silo_dataset.csv)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = args.output or config["paths"]["raw_dataset"]

    print("[SMISIA] Generando dataset sintético...")
    dataset = generate_dataset(config)

    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    dataset.to_csv(output_path, index=False)

    # Estadísticas
    print(f"[SMISIA] Dataset generado: {output_path}")
    print(f"[SMISIA] Total registros: {len(dataset):,}")
    print(f"[SMISIA] Silos: {dataset['silo_id'].nunique()}")
    print(f"[SMISIA] Rango temporal: {dataset['timestamp'].min()} — {dataset['timestamp'].max()}")
    print("[SMISIA] Distribución de labels:")
    label_counts = dataset["label"].value_counts()
    for label, count in label_counts.items():
        pct = 100 * count / len(dataset)
        print(f"         {label}: {count:,} ({pct:.1f}%)")
    print("[SMISIA] Missing values (NaN):")
    for col in ["temperature_c", "humidity_pct", "co2_ppm", "nh3_ppm"]:
        n_missing = dataset[col].isna().sum()
        pct = 100 * n_missing / len(dataset)
        print(f"         {col}: {n_missing:,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
