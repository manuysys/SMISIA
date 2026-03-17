"""
SMISIA — Módulo de Insights y Analítica Avanzada
Implementa Features 1-10 para inteligencia de almacenamiento.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger("smisia.insights")

# --- FEATURE 1 & 2: Detección de Silos Riesgosos ---

def get_risk_score(silo_data: dict) -> float:
    """Calcula el score de riesgo (problema + critico)."""
    raw = silo_data.get("raw_scores", {})
    return float(raw.get("problema", 0) + raw.get("critico", 0))

def get_worst_silo(all_silos: Dict[str, dict]) -> Tuple[Optional[str], float]:
    """Identifica el silo con mayor riesgo acumulado."""
    if not all_silos:
        return None, 0.0
    
    worst_id = None
    max_risk = -1.0
    
    for sid, data in all_silos.items():
        risk = get_risk_score(data)
        if risk > max_risk:
            max_risk = risk
            worst_id = sid
            
    return worst_id, max_risk

def get_top_risky_silos(all_silos: Dict[str, dict], k: int = 3) -> List[Tuple[str, float]]:
    """Devuelve el Top K de silos ordenados por riesgo."""
    risks = [(sid, get_risk_score(data)) for sid, data in all_silos.items()]
    risks.sort(key=lambda x: x[1], reverse=True)
    return risks[:k]


# --- FEATURE 3: Deterioro más Rápido ---

def get_fastest_deteriorating_silo(all_silos: Dict[str, dict]) -> Tuple[Optional[str], float]:
    """
    Calcula el silo que empeora más rápido basado en tendencias.
    Score = 0.5 * hum_trend + 0.3 * co2_trend + 0.2 * temp_trend
    """
    if not all_silos:
        return None, 0.0
    
    fastest_id = None
    max_score = -100.0 # Las tendencias pueden ser negativas
    
    for sid, data in all_silos.items():
        trends = data.get("trend", {})
        # Normalizamos un poco basándonos en esperados de escala (p.p. vs ppm vs °C)
        h_t = trends.get("humidity_pct", 0)
        c_t = trends.get("co2_ppm", 0) / 100.0 # Normalizar CO2
        t_t = trends.get("temperature_c", 0)
        
        det_score = (0.5 * h_t) + (0.3 * c_t) + (0.2 * t_t)
        
        if det_score > max_score:
            max_score = det_score
            fastest_id = sid
            
    return fastest_id, max_score


# --- FEATURE 4 & 5: Salud de Sensores ---

def detect_sensor_anomalies(sensor_df: pd.DataFrame) -> List[str]:
    """Detecta comportamientos sospechosos en los sensores."""
    alerts = []
    if sensor_df.empty: return []
    
    # 1. Valores imposibles o extremos
    if "humidity_pct" in sensor_df.columns:
        if (sensor_df["humidity_pct"] > 35).any() or (sensor_df["humidity_pct"] < 5).any():
            alerts.append("Sensor de humedad con valores fuera de rango físico.")
            
    # 2. Picos súbitos (Spikes)
    diffs = sensor_df.select_dtypes(include=[np.number]).diff().abs()
    for col in diffs.columns:
        # Si cambia más de un 20% del rango normal en una lectura
        if (diffs[col] > 3 * sensor_df[col].std()).any(): # Umbral heurístico
            alerts.append(f"Posible salto súbito (spike) en sensor de {col}.")
            
    # 3. Datos planos (Sensor trabado)
    for col in sensor_df.select_dtypes(include=[np.number]).columns:
        if len(sensor_df) > 5 and sensor_df[col].std() < 1e-6:
            alerts.append(f"Lectura plana detectada en {col}. El sensor podría estar trabado.")
            
    return alerts

def get_sensor_health_score(sensor_df: pd.DataFrame) -> float:
    """Calcula confiabilidad de 0 a 1."""
    if sensor_df.empty: return 0.0
    
    # Penalización por missing values
    missing_ratio = sensor_df.isnull().sum().sum() / sensor_df.size
    
    # Penalización por inestabilidad de señal (RSSI/SNR si existen)
    signal_penalty = 0.0
    if "rssi" in sensor_df.columns:
        rssi_std = sensor_df["rssi"].std()
        if rssi_std > 10: signal_penalty += 0.2
        
    health = 1.0 - (missing_ratio * 0.5) - signal_penalty
    return max(0.0, min(1.0, health))


# --- FEATURE 6 & 10: Salud Global e Insights ---

def get_global_storage_health(all_silos: Dict[str, dict]) -> Tuple[float, str]:
    """Salud general de todo el almacenamiento."""
    if not all_silos: return 1.0, "Sin datos"
    
    risks = [get_risk_score(d) for d in all_silos.values()]
    avg_risk = np.mean(risks)
    health_score = 1.0 - avg_risk
    
    status = "Estable"
    if health_score < 0.6: status = "Riesgo Elevado"
    elif health_score < 0.85: status = "Atención Requerida"
    
    return health_score, status

def detect_global_risk_pattern(all_silos: Dict[str, dict]) -> List[str]:
    """Detecta si hay problemas sistemáticos."""
    insights = []
    rising_hum = 0
    rising_co2 = 0
    total = len(all_silos)
    
    for d in all_silos.values():
        trends = d.get("trend", {})
        if trends.get("humidity_pct", 0) > 0.2: rising_hum += 1
        if trends.get("co2_ppm", 0) > 50: rising_co2 += 1
        
    if rising_hum > total / 2 and total > 1:
        insights.append(f"Patrón detectado: Tendencia de humedad creciente en {rising_hum} silobolsas.")
    if rising_co2 > 2:
        insights.append(f"Alerta: Nivel de CO2 en aumento en {rising_co2} unidades. Posible actividad biológica global.")
        
    return insights


# --- FEATURE 7: Incertidumbre ---

def check_model_uncertainty(confidence: float) -> Optional[str]:
    """Avisa si la confianza es baja."""
    if confidence < 0.55:
        return "⚠️ Predicción incierta (confianza baja). Se recomienda monitoreo físico adicional."
    return None


# --- FEATURE 8: Escalamiento de Riesgo ---

def detect_risk_escalation(silo_id: str, current_status: str, history: List[dict]) -> Optional[str]:
    """Detecta si el riesgo subió recientemente."""
    if not history: return None
    
    last_status = history[-1].get("status")
    order = {"bien": 0, "tolerable": 1, "problema": 2, "critico": 3}
    
    curr_rank = order.get(current_status, 0)
    prev_rank = order.get(last_status, 0)
    
    if curr_rank > prev_rank:
        return f"🚨 Escalamiento detectado: El silo {silo_id} pasó de {last_status.upper()} a {current_status.upper()}."
    return None


# --- FEATURE 9: Resumen de Tendencia Humano ---

def generate_trend_summary(silo_data: dict) -> str:
    """Traduce tendencias a frases naturales."""
    trends = silo_data.get("trend", {})
    hum = trends.get("humidity_pct", 0)
    co2 = trends.get("co2_ppm", 0)
    
    phrases = []
    if abs(hum) > 0.1:
        verb = "aumentó" if hum > 0 else "disminuyó"
        phrases.append(f"La humedad {verb} un {abs(hum):.1f}% en las últimas 24h.")
        
    if co2 > 40:
        phrases.append("Los niveles de CO2 están subiendo rápidamente, sugiriendo respiración del grano.")
        
    return " ".join(phrases) if phrases else "Tendencias estables en el último periodo."
