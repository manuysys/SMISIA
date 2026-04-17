# AGRILION 🌾

## Sistema ML de Análisis de Silobolsas Agrícolas

AGRILION es un sistema completo de Machine Learning para monitoreo y análisis predictivo de silobolsas agrícolas. Utiliza datos de sensores (temperatura, humedad, CO2) recolectados vía LoRa para detectar riesgos de deterioro del grano antes de que generen pérdidas.

---

## 📋 Características

- **Series Temporales**: Análisis de datos de sensores con LSTM
- **Detección de Anomalías**: 3 métodos (Z-Score, Moving Average, Isolation Forest) con consenso
- **Predicción**: Forecasting single-step y multi-step con LSTM
- **Scoring de Riesgo**: Motor ponderado (0-100) con reglas agronómicas
- **Alertas Inteligentes**: Alertas contextualizadas en español con recomendaciones
- **API REST**: FastAPI con endpoints de predicción y análisis
- **Visualización**: Gráficos profesionales con estilo dark theme

---

## 🏗️ Estructura del Proyecto

```
agrilion/
├── src/                         # Paquete principal
│   ├── __init__.py
│   ├── config.py                # Configuración centralizada
│   ├── data_loader.py           # Carga y limpieza de datos
│   ├── preprocessing.py         # Normalización y windowing LSTM
│   ├── anomaly_detection.py     # Detección de anomalías (3 métodos)
│   ├── lstm_model.py            # Modelo LSTM (TensorFlow/Keras)
│   ├── predictor.py             # Predicción y evaluación
│   ├── risk_engine.py           # Motor de riesgo agronómico
│   ├── alerts.py                # Sistema de alertas inteligentes
│   └── visualization.py         # Gráficos profesionales
├── api/                         # FastAPI
│   ├── __init__.py
│   ├── app.py                   # Aplicación FastAPI
│   ├── schemas.py               # Modelos Pydantic
│   └── routes.py                # Endpoints REST
├── data/                        # Datos
│   └── synthetic_generator.py   # Generador de datos sintéticos
├── models/                      # Modelos entrenados (.keras)
├── outputs/                     # Gráficos generados
├── tests/                       # Tests unitarios
│   └── test_pipeline.py
├── main.py                      # Pipeline completo
├── requirements.txt             # Dependencias
└── README.md
```

---

## 🚀 Instalación

```bash
# Clonar o navegar al proyecto
cd src

# Crear entorno virtual (recomendado)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

---

## ▶️ Uso Rápido

### Pipeline Completo
```bash
# Ejecutar pipeline end-to-end (genera datos sintéticos si no existen)
python main.py

# Con datos propios
python main.py --data path/to/sensor_data.csv

# Personalizar entrenamiento
python main.py --epochs 100 --batch-size 64

# Usar modelo ya entrenado (sin re-entrenar)
python main.py --skip-training

# Sin gráficos
python main.py --no-plots
```

### API REST
```bash
# Iniciar la API
uvicorn api.app:app --reload --port 8000

# Documentación interactiva
# http://localhost:8000/docs
```

### Endpoints de la API

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET`  | `/api/v1/health` | Health check |
| `GET`  | `/api/v1/model/status` | Estado del modelo |
| `POST` | `/api/v1/predict` | Predicción de valores futuros |
| `POST` | `/api/v1/analyze` | Análisis completo (anomalías + riesgo + alertas) |

#### Ejemplo: Predicción
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "readings": [...],  # 24+ lecturas de sensores
    "steps": 6          # Pasos futuros a predecir
  }'
```

---

## 🧠 Arquitectura del Modelo LSTM

```
Input(24 timesteps, 3 features)
    → LSTM(128, return_sequences=True)
    → Dropout(0.2)
    → LSTM(64)
    → Dropout(0.2)
    → Dense(32, ReLU)
    → Dense(3)  # Predicción multivariable
```

- **Ventana de entrada**: 24 mediciones (~24 horas)
- **Optimizador**: Adam (lr=0.001)
- **Loss**: MSE
- **Callbacks**: EarlyStopping + ReduceLROnPlateau

---

## 🔍 Detección de Anomalías

| Método | Tipo | Parámetros |
|--------|------|------------|
| **Z-Score** | Estadístico | threshold=3.0σ |
| **Moving Average** | Tendencia local | window=12, threshold=2.0σ |
| **Isolation Forest** | ML no supervisado | contamination=5% |

**Consenso**: Una anomalía se confirma si ≥2 de 3 métodos la detectan.

---

## ⚠️ Motor de Riesgo

### Factores Ponderados

| Factor | Peso | Lógica |
|--------|------|--------|
| Humedad + Temperatura | 30% | Sinergia hongos/bacterias |
| CO2 | 25% | Actividad biológica / fermentación |
| Anomalías estadísticas | 20% | Proporción en ventana reciente |
| Desviación LSTM | 25% | Error de predicción normalizado |

### Niveles de Riesgo

| Score | Nivel | Significado |
|-------|-------|-------------|
| 0–30 | 🟢 NORMAL | Operación segura |
| 30–70 | 🟡 WARNING | Monitoreo recomendado |
| 70–100 | 🔴 CRITICAL | Acción urgente requerida |

---

## 📊 Gráficos Generados

1. **Series Temporales**: Visualización completa de sensores
2. **Predicciones vs Reales**: Overlay con métricas R²/MAE
3. **Anomalías Detectadas**: Series con puntos anómalos marcados
4. **Timeline de Riesgo**: Zonas de color por nivel
5. **Historial de Entrenamiento**: Loss y MAE por época

---

## 🧪 Tests

```bash
# Ejecutar todos los tests
python -m pytest tests/ -v

# Tests específicos
python -m pytest tests/test_pipeline.py::TestRiskEngine -v
```

---

## 📡 Formato de Datos

### CSV de Entrada
```csv
timestamp,temperature,humidity,co2,silo_id
2025-01-15 00:00:00,24.5,62.3,418.0,SILO_001
2025-01-15 01:00:00,23.8,64.1,422.5,SILO_001
...
```

### Columnas Requeridas
- `timestamp`: Fecha y hora (ISO 8601)
- `temperature`: Temperatura en °C
- `humidity`: Humedad relativa en %
- `co2`: Dióxido de carbono en ppm

---

## 🔧 Configuración

Todos los parámetros se centralizan en `src/config.py`:
- Umbrales agronómicos
- Arquitectura LSTM
- Parámetros de detección de anomalías
- Pesos del motor de riesgo
- Configuración de visualización

---

## 📄 Licencia

Proyecto privado — AGRILION © 2025
