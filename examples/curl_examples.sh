#!/bin/bash
# ========================================================
# SMISIA — Ejemplos de uso con curl
# ========================================================

API_URL="http://localhost:8000"
API_KEY="smisia-dev-key-2026"

echo "=========================================="
echo "  SMISIA — Ejemplos curl"
echo "=========================================="

# 1. Health check (sin auth)
echo ""
echo "--- 1. Health Check ---"
curl -s "$API_URL/metrics" | python -m json.tool

# 2. Inferencia individual
echo ""
echo "--- 2. Inferencia Individual ---"
curl -s -X POST "$API_URL/infer" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "silo_id": "SILO_001",
    "timestamp": "2025-12-01T14:00:00-03:00",
    "fill_date": "2025-10-01T00:00:00-03:00",
    "recent_readings": [
      {
        "timestamp": "2025-12-01T10:00:00-03:00",
        "temperature_c": 26.5,
        "humidity_pct": 14.2,
        "co2_ppm": 800,
        "nh3_ppm": 5.0,
        "battery_pct": 78,
        "rssi": -82,
        "snr": 9.5
      },
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
  }' | python -m json.tool

# 3. Status de un silo
echo ""
echo "--- 3. Status ---"
curl -s -H "X-API-Key: $API_KEY" "$API_URL/status/SILO_001" | python -m json.tool

# 4. Explicaciones
echo ""
echo "--- 4. Explicaciones ---"
curl -s -H "X-API-Key: $API_KEY" "$API_URL/explain/SILO_001" | python -m json.tool

# 5. Chatbot
echo ""
echo "--- 5. Chatbot ---"
curl -s -X POST "$API_URL/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "message": "¿Cuál es el estado de la silobolsa SILO_001?",
    "silo_id": "SILO_001"
  }' | python -m json.tool

# 6. Chatbot - predicción
echo ""
echo "--- 6. Chatbot - Predicción ---"
curl -s -X POST "$API_URL/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "message": "¿Va a empeorar en 3 días?",
    "silo_id": "SILO_001"
  }' | python -m json.tool

# 7. Batch inference
echo ""
echo "--- 7. Batch Inference ---"
curl -s -X POST "$API_URL/batch_infer" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "items": [
      {
        "silo_id": "SILO_001",
        "timestamp": "2025-12-01T14:00:00-03:00",
        "recent_readings": [{"timestamp": "2025-12-01T14:00:00-03:00", "temperature_c": 25, "humidity_pct": 14, "co2_ppm": 500, "battery_pct": 90, "rssi": -80, "snr": 10}]
      },
      {
        "silo_id": "SILO_002",
        "timestamp": "2025-12-01T14:00:00-03:00",
        "recent_readings": [{"timestamp": "2025-12-01T14:00:00-03:00", "temperature_c": 35, "humidity_pct": 22, "co2_ppm": 1800, "battery_pct": 45, "rssi": -95, "snr": 5}]
      }
    ]
  }' | python -m json.tool

echo ""
echo "=========================================="
echo "  Ejemplos completados"
echo "=========================================="
