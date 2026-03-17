# SMISIA — Sistema de Monitoreo e IA para Silobolsas

Sistema completo de monitoreo inteligente para silobolsas que combina sensores IoT con modelos de machine learning para clasificar el estado del grano, predecir deterioros y detectar anomalías.

## 🏗️ Arquitectura

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Sensores    │────▶│  FastAPI      │────▶│  Modelos ML  │
│  (LoRa/IoT)  │     │  REST API    │     │  XGBoost     │
└──────────────┘     └──────┬───────┘     │  LSTM        │
                           │              │  IsoForest   │
                    ┌──────▼───────┐     └──────────────┘
                    │  Chatbot     │
                    │  (español)   │
                    └──────────────┘
```

## 📋 Características

- **Clasificación en 4 estados**: bien ✅ | tolerable ⚠️ | problema 🚨 | crítico 🔴
- **Predicción temporal**: horizonte 1-7 días con LSTM bidireccional
- **Detección de anomalías**: Isolation Forest entrenado sobre datos normales
- **Explicabilidad**: SHAP + top-3 drivers por inferencia
- **Estimación de incertidumbre**: ensemble bootstrap + MC Dropout
- **Chatbot en español**: respuestas naturales con emojis y recomendaciones
- **API REST completa**: FastAPI con docs auto-generadas

## 🚀 Inicio Rápido

### Requisitos
- Python 3.11+
- Docker y Docker Compose (para despliegue)

### Instalación local

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Generar dataset sintético
python generate_dataset.py

# 3. Entrenar modelos
python train.py

# 4. Evaluar métricas
python eval.py

# 5. Servir API
python serve.py
```

### Despliegue con Docker

```bash
# Opción 1: Script automático
bash deploy.sh

# Opción 2: Manual
docker-compose up -d
```

La API estará disponible en `http://localhost:8000`

## 📡 API Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/infer` | Inferencia individual |
| GET | `/status/{silo_id}` | Último estado inferido |
| POST | `/batch_infer` | Inferencia en lote |
| GET | `/explain/{silo_id}` | Explicaciones SHAP |
| GET | `/metrics` | Salud del servicio |
| POST | `/chat` | Chatbot en español |

**Autenticación**: Header `X-API-Key: smisia-dev-key-2026`

**Documentación interactiva**: `http://localhost:8000/docs`

### Ejemplo de uso

```bash
# Inferencia
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -H "X-API-Key: smisia-dev-key-2026" \
  -d '{
    "silo_id": "SILO_001",
    "timestamp": "2025-12-01T14:00:00-03:00",
    "recent_readings": [{
      "timestamp": "2025-12-01T14:00:00-03:00",
      "temperature_c": 28.4,
      "humidity_pct": 16.8,
      "co2_ppm": 1350,
      "battery_pct": 62,
      "rssi": -85,
      "snr": 8.5
    }]
  }'

# Estado
curl -H "X-API-Key: smisia-dev-key-2026" \
  http://localhost:8000/status/SILO_001

# Chatbot
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: smisia-dev-key-2026" \
  -d '{"message": "¿Cuál es el estado de la silobolsa SILO_001?", "silo_id": "SILO_001"}'
```

## 📁 Estructura del Proyecto

```
├── config.yml                 # Configuración central
├── generate_dataset.py        # Generador de datos sintéticos
├── train.py                   # Pipeline de entrenamiento
├── eval.py                    # Pipeline de evaluación
├── serve.py                   # Servidor API
├── requirements.txt           # Dependencias Python
├── Dockerfile                 # Imagen Docker
├── docker-compose.yml         # Orquestación de servicios
├── deploy.sh                  # Script de despliegue
├── src/
│   ├── config.py             # Carga de configuración
│   ├── preprocessing/        # Limpieza y validación
│   ├── features/             # Feature engineering
│   ├── labeling/             # Etiquetado heurístico
│   ├── models/               # XGBoost, LSTM, anomalías
│   ├── api/                  # FastAPI REST
│   └── chatbot/              # Bot en español
├── tests/                    # Tests unitarios e integración
├── models/                   # Modelos serializados
├── data/                     # Datasets
├── docs/                     # Documentación adicional
└── examples/                 # Ejemplos de uso
```

## 🧪 Tests

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Con cobertura
pytest tests/ --cov=src --cov-report=term-missing

# Test específico
pytest tests/test_api.py -v
```

## 📊 Modelos

### Fase A: XGBoost (Clasificación)
- 4 clases: bien / tolerable / problema / crítico
- TimeSeriesSplit con 5 folds
- SHAP TreeExplainer para explicabilidad
- Ensemble de 5 modelos bootstrap para incertidumbre

### Fase B: LSTM (Predicción Temporal)
- Bidireccional, 2 capas (128/64 unidades)
- Horizontes: 1, 3, 7 días
- MC Dropout (T=20) para incertidumbre

### Fase C: Isolation Forest (Anomalías)
- Entrenado sobre datos "bien"
- Score normalizado 0-1

## ⚙️ Configuración

Toda la configuración se centraliza en `config.yml`:
- Umbrales de alertas
- Ventanas de features (6h, 24h, 72h, 7d)
- Hiperparámetros de modelos
- Configuración de API/seguridad

## 📄 Licencia

Proyecto interno — SMISIA 2026.
