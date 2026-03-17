# SMISIA — Especificación de API

## Base URL
```
http://localhost:8000
```

## Autenticación
Todas las rutas excepto `/metrics`, `/docs`, y `/redoc` requieren el header:
```
X-API-Key: smisia-dev-key-2026
```

## Endpoints

### POST /infer
Realiza inferencia sobre una silobolsa individual.

**Request Body:**
```json
{
  "silo_id": "SILO_001",
  "timestamp": "2025-12-01T14:00:00-03:00",
  "fill_date": "2025-10-01T00:00:00-03:00",
  "recent_readings": [
    {
      "timestamp": "2025-12-01T12:00:00-03:00",
      "temperature_c": 27.5,
      "humidity_pct": 15.2,
      "co2_ppm": 1100,
      "nh3_ppm": 8.5,
      "battery_pct": 75,
      "rssi": -82,
      "snr": 9.1
    },
    {
      "timestamp": "2025-12-01T14:00:00-03:00",
      "temperature_c": 28.4,
      "humidity_pct": 16.8,
      "co2_ppm": 1350,
      "battery_pct": 62,
      "rssi": -85,
      "snr": 8.5
    }
  ]
}
```

**Response (200):**
```json
{
  "silo_id": "SILO_001",
  "timestamp": "2025-12-01T14:00:00-03:00",
  "status": "problema",
  "confidence": 0.78,
  "uncertainty_std": 0.06,
  "summary": "Humedad y/o temperatura en aumento; CO₂ elevado.",
  "metrics": {
    "temperature_c": {"value": 28.4, "unit": "°C"},
    "humidity_pct": {"value": 16.8, "unit": "%"},
    "co2_ppm": {"value": 1350.0, "unit": "ppm"},
    "battery_pct": {"value": 62.0, "unit": "%"}
  },
  "trend": {
    "humidity_pct_delta_24h": 1.8,
    "temperature_c_delta_24h": 0.9
  },
  "explanations": [
    {"feature": "humidity_delta_72h", "impact": 0.42},
    {"feature": "co2_level", "impact": 0.27},
    {"feature": "temperature_slope", "impact": 0.15}
  ],
  "recommended_action": "Inspección física y ventilación si es posible.",
  "raw_scores": {
    "bien": 0.05,
    "tolerable": 0.12,
    "problema": 0.78,
    "critico": 0.05
  },
  "anomaly_score": 0.62
}
```

### GET /status/{silo_id}
Devuelve el último estado inferido (cache en memoria).

### POST /batch_infer
Inferencia en lote. Acepta lista de items con el mismo formato que `/infer`.

### GET /explain/{silo_id}
Devuelve explicaciones SHAP y top drivers.

### GET /metrics
Health check y métricas del servicio (no requiere auth).

### POST /chat
Chatbot en español. Acepta mensajes naturales y devuelve respuestas formateadas.

## Documentación interactiva
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
