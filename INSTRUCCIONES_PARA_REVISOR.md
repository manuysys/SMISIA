# INSTRUCCIONES PARA REVISOR

## Requisitos previos
- Python 3.11+ instalado
- pip actualizado
- (Opcional) Docker y Docker Compose para despliegue containerizado

## Pasos para reproducir

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Generar dataset sintético
```bash
python generate_dataset.py
```
**Salida esperada**: `data/synthetic_silo_dataset.csv` con ~72,000 registros.

### 3. Entrenar modelos
```bash
# Entrenamiento completo (XGBoost + LSTM + Anomaly)
python train.py

# Solo XGBoost (más rápido, recomendado para primera revisión)
python train.py --skip-lstm
```
**Salida esperada**: Modelos serializados en `models/`:
- `xgboost_model.joblib` — clasificador principal
- `bootstrap_models.joblib` — ensemble para incertidumbre
- `feature_columns.joblib` — columnas de features
- `feature_importance.joblib` — importancia de features
- `anomaly_model.joblib` — detector de anomalías
- `anomaly_scaler.joblib` — scaler para anomalías
- `lstm_model.pt` — modelo temporal (si no se omitió)

### 4. Evaluar métricas
```bash
python eval.py
```
**Salida esperada**: Reporte en consola + `models/evaluation_report.json` con:
- Confusion matrix
- Precision/Recall/F1 por clase
- AUC por clase
- Tests de robustez (ruido + missingness)

### 5. Servir la API
```bash
python serve.py
```
Abrir en navegador: `http://localhost:8000/docs`

### 6. Probar la API
```bash
# Health check (sin auth)
curl http://localhost:8000/metrics

# Inferencia (con auth)
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
```

### 7. Ejecutar tests
```bash
pytest tests/ -v --tb=short
```

## Dónde ver los artefactos

| Artefacto | Ubicación |
|-----------|-----------|
| Dataset generado | `data/synthetic_silo_dataset.csv` |
| Dataset procesado | `data/feature_dataset.parquet` |
| Modelos serializados | `models/` |
| Reporte de evaluación | `models/evaluation_report.json` |
| Documentación API | `http://localhost:8000/docs` |
| Configuración | `config.yml` |

## Validación de métricas

Las métricas objetivo son:
- **Recall "crítico" ≥ 0.85** — verificar en `evaluation_report.json`
- **Macro F1** — reportado en evaluación
- **AUC por clase** — reportado en evaluación
- **Robustez**: F1 con 20% ruido y 50% missing data

## Despliegue con Docker
```bash
bash deploy.sh
# O manualmente:
docker-compose up -d
```

La API estará en `http://localhost:8000`.
