"""
AGRILION — Generador de Datos Sintéticos
==========================================

Genera datos realistas de sensores agrícolas para silobolsas.
Incluye patrones diurnos, tendencias, ruido y eventos anómalos
controlados para validar la detección.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Agregar el directorio padre al path para importar config
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import SYNTHETIC_CONFIG, DEFAULT_CSV_PATH


def generate_synthetic_data(
    days: int = None,
    interval_hours: float = None,
    seed: int = None,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Genera datos sintéticos realistas de sensores para silobolsas.

    Los datos incluyen:
    - Ciclos diurnos (temperatura más alta de día, humedad más alta de noche)
    - Correlación parcial inversa entre temperatura y humedad
    - CO2 base estable con variaciones naturales
    - Tendencia gradual ascendente (simulando deterioro lento)
    - Eventos anómalos inyectados (spikes de temperatura, humedad, CO2)

    Parameters
    ----------
    days : int, optional
        Número de días a generar. Default: config.
    interval_hours : float, optional
        Intervalo entre mediciones en horas. Default: config.
    seed : int, optional
        Semilla aleatoria. Default: config.
    output_path : str, optional
        Ruta para guardar el CSV. Default: config.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas: timestamp, temperature, humidity, co2, silo_id
    """
    # Parámetros
    days = days or SYNTHETIC_CONFIG["days"]
    interval_hours = interval_hours or SYNTHETIC_CONFIG["interval_hours"]
    seed = seed or SYNTHETIC_CONFIG["random_seed"]
    output_path = Path(output_path) if output_path else DEFAULT_CSV_PATH

    np.random.seed(seed)

    # Generar timestamps
    total_hours = days * 24
    n_points = int(total_hours / interval_hours)
    start_date = pd.Timestamp("2025-01-15 00:00:00")
    timestamps = pd.date_range(start=start_date, periods=n_points, freq=f"{int(interval_hours * 60)}min")

    # Índice horario para ciclos diurnos (0-23)
    hours = np.array([t.hour + t.minute / 60.0 for t in timestamps])

    # ==========================================================================
    # TEMPERATURA
    # ==========================================================================
    temp_cfg = SYNTHETIC_CONFIG["temperature"]

    # Ciclo diurno: máximo ~14h, mínimo ~5h
    temp_diurnal = temp_cfg["diurnal_amplitude"] * np.sin(
        2 * np.pi * (hours - 5) / 24
    )

    # Tendencia gradual ascendente (simula deterioro lento)
    trend = np.linspace(0, 3.0, n_points)

    # Ruido gaussiano
    noise = np.random.normal(0, temp_cfg["base_std"], n_points)

    # Variación día a día (macro ruido)
    day_variation = np.repeat(
        np.random.normal(0, 1.5, days),
        int(24 / interval_hours)
    )[:n_points]

    temperature = temp_cfg["base_mean"] + temp_diurnal + trend + noise + day_variation

    # ==========================================================================
    # HUMEDAD
    # ==========================================================================
    hum_cfg = SYNTHETIC_CONFIG["humidity"]

    # Ciclo diurno INVERSO a temperatura (más humedad de noche)
    hum_diurnal = hum_cfg["diurnal_amplitude"] * np.sin(
        2 * np.pi * (hours - 5) / 24 + np.pi  # Desfasado 180°
    )

    # Correlación parcial inversa con temperatura
    temp_influence = -0.3 * (temperature - temp_cfg["base_mean"])

    # Tendencia ascendente leve
    hum_trend = np.linspace(0, 5.0, n_points)

    noise_h = np.random.normal(0, hum_cfg["base_std"], n_points)
    day_var_h = np.repeat(
        np.random.normal(0, 2.0, days),
        int(24 / interval_hours)
    )[:n_points]

    humidity = hum_cfg["base_mean"] + hum_diurnal + temp_influence + hum_trend + noise_h + day_var_h

    # Clip a rangos físicos
    humidity = np.clip(humidity, 20.0, 98.0)

    # ==========================================================================
    # CO2
    # ==========================================================================
    co2_cfg = SYNTHETIC_CONFIG["co2"]

    # Variación diurna leve
    co2_diurnal = co2_cfg["diurnal_amplitude"] * np.sin(
        2 * np.pi * (hours - 3) / 24
    )

    # Tendencia ascendente más pronunciada (deterioro genera CO2)
    co2_trend = np.linspace(0, 80.0, n_points)

    # Ruido y autocorrelación (CO2 cambia más lentamente)
    noise_c = np.zeros(n_points)
    noise_c[0] = np.random.normal(0, co2_cfg["base_std"])
    for i in range(1, n_points):
        noise_c[i] = 0.8 * noise_c[i - 1] + np.random.normal(0, co2_cfg["base_std"] * 0.5)

    co2 = co2_cfg["base_mean"] + co2_diurnal + co2_trend + noise_c

    # ==========================================================================
    # INYECCIÓN DE ANOMALÍAS
    # ==========================================================================
    n_anomalies = SYNTHETIC_CONFIG["n_anomaly_events"]
    anomaly_duration = int(SYNTHETIC_CONFIG["anomaly_duration_hours"] / interval_hours)

    # Seleccionar posiciones aleatorias para anomalías (evitando bordes)
    margin = anomaly_duration * 2
    anomaly_starts = np.random.choice(
        range(margin, n_points - margin),
        size=n_anomalies,
        replace=False,
    )
    anomaly_starts.sort()

    # Tipos de anomalía
    anomaly_types = [
        "heat_spike",        # Calentamiento repentino
        "humidity_surge",    # Subida de humedad
        "co2_fermentation",  # CO2 por fermentación
        "combined_risk",     # Múltiples sensores
        "cold_shock",        # Descenso brusco de temperatura
    ]

    anomaly_labels = np.full(n_points, "normal", dtype=object)

    for i, start in enumerate(anomaly_starts):
        end = min(start + anomaly_duration, n_points)
        atype = anomaly_types[i % len(anomaly_types)]

        # Forma de campana para la anomalía (más natural)
        window_len = end - start
        bell = np.exp(-0.5 * ((np.linspace(-2, 2, window_len)) ** 2))

        if atype == "heat_spike":
            temperature[start:end] += temp_cfg["anomaly_spike"] * bell
            co2[start:end] += 100 * bell  # El calor genera algo de CO2
            anomaly_labels[start:end] = "heat_spike"

        elif atype == "humidity_surge":
            humidity[start:end] += hum_cfg["anomaly_spike"] * bell
            anomaly_labels[start:end] = "humidity_surge"

        elif atype == "co2_fermentation":
            co2[start:end] += co2_cfg["anomaly_spike"] * bell
            temperature[start:end] += 4.0 * bell  # Fermentación genera calor
            anomaly_labels[start:end] = "co2_fermentation"

        elif atype == "combined_risk":
            temperature[start:end] += temp_cfg["anomaly_spike"] * 0.7 * bell
            humidity[start:end] += hum_cfg["anomaly_spike"] * 0.8 * bell
            co2[start:end] += co2_cfg["anomaly_spike"] * 0.6 * bell
            anomaly_labels[start:end] = "combined_risk"

        elif atype == "cold_shock":
            temperature[start:end] -= 15.0 * bell
            anomaly_labels[start:end] = "cold_shock"

    # Clip final a rangos válidos
    temperature = np.clip(temperature, -5.0, 55.0)
    humidity = np.clip(humidity, 10.0, 99.0)
    co2 = np.clip(co2, 250.0, 3000.0)

    # ==========================================================================
    # DATOS FALTANTES (realismo)
    # ==========================================================================
    # Insertar ~1% de NaN aleatorios
    missing_mask = np.random.random(n_points) < 0.01
    temperature_out = temperature.copy()
    humidity_out = humidity.copy()
    co2_out = co2.copy()
    temperature_out[missing_mask] = np.nan
    humidity_out[missing_mask] = np.nan
    co2_out[missing_mask] = np.nan

    # ==========================================================================
    # CONSTRUIR DATAFRAME
    # ==========================================================================
    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": np.round(temperature_out, 2),
        "humidity": np.round(humidity_out, 2),
        "co2": np.round(co2_out, 2),
        "silo_id": SYNTHETIC_CONFIG["silo_id"],
        "anomaly_label": anomaly_labels,  # Ground truth para validación
    })

    # Guardar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Datos sintéticos generados: {len(df)} registros ({days} días)")
    print(f"   📁 Guardados en: {output_path}")
    print(f"   🔴 Anomalías inyectadas: {n_anomalies} eventos")
    print(f"   📊 Rango de fechas: {timestamps[0]} → {timestamps[-1]}")

    # Estadísticas
    print(f"\n   📈 Estadísticas:")
    for col in ["temperature", "humidity", "co2"]:
        valid = df[col].dropna()
        print(f"      {col:>12}: μ={valid.mean():.1f}, σ={valid.std():.1f}, "
              f"min={valid.min():.1f}, max={valid.max():.1f}")

    return df


if __name__ == "__main__":
    df = generate_synthetic_data()
    print(f"\n{df.head(10)}")
