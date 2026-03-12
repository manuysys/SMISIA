# LIMITACIONES

## Datos Sintéticos

⚠️ **Todo el sistema fue entrenado exclusivamente con datos sintéticos generados por `generate_dataset.py`.**

- Los patrones simulados son aproximaciones a dinámicas reales de silobolsas, pero **no reemplazan datos de sensores reales**.
- La distribución de clases fue ajustada manualmente (~70% bien, 15% tolerable, 10% problema, 5% crítico).
- El rendimiento del modelo en producción **será diferente** al reportado con datos sintéticos.
- Se recomienda **reentrenar los modelos** con datos reales antes de usar en producción.

## Umbrales Heurísticos

Los umbrales de clasificación son valores estimados basados en literatura agrícola:

| Parámetro | Valor usado | Nota |
|-----------|-------------|------|
| Humedad crítica | > 25% | Varía según tipo de grano |
| CO₂ crítico | > 2000 ppm | Depende del tiempo de exposición |
| Temperatura crítica | > 45°C | Indicador de fermentación |
| Humedad problema | 18–25% | Zona gris, requiere validación |
| CO₂ problema | 1200–2000 ppm | Sostenido por 48h |

**Estos umbrales deben ser calibrados con agrónomos y datos reales.**

## Sensores Faltantes

- **NH₃ (amoníaco)**: Marcado como opcional. Si no hay lecturas de NH₃, el modelo lo ignora pero la precisión puede disminuir para detección temprana de deterioro proteico.
- **Si falta CO₂**: Se pierde un indicador clave de fermentación. El modelo se apoyará más en temperatura y humedad, reduciendo la capacidad de detección temprana.
- **RSSI/SNR**: Usados para evaluar calidad de señal, no para clasificación directa.

## Modelo LSTM

- El LSTM requiere secuencias largas (hasta 30 días). Para silobolsas con datos de menos de 30 días, la predicción temporal será menos confiable.
- MC Dropout es una aproximación a la incertidumbre bayesiana, no incertidumbre exacta.
- El TFT (Temporal Fusion Transformer) fue omitido por complejidad. Se recomienda evaluarlo con datasets más grandes.

## API y Despliegue

- La autenticación es por **API key simple**. Para producción, implementar OAuth2/JWT.
- El cache de resultados es **in-memory** (no persistente). Se reinicia con la API.
- Redis está configurado pero la integración de cache Redis no está implementada en esta versión.
- PostgreSQL está en docker-compose pero no se usa activamente para almacenamiento de inferencias.

## Rendimiento

- La inferencia XGBoost es rápida (< 100ms), pero el preprocesamiento y feature engineering pueden agregar latencia significativa con muchas lecturas históricas.
- El LSTM requiere GPU para inferencia rápida. En CPU, puede tardar > 2s con secuencias largas.

## Active Learning

- El pipeline de active learning es un **stub**: selecciona muestras inciertas pero no tiene UI para revisión humana.
- La "revisión humana" debe implementarse como una interfaz web separada.

## MLflow / DVC

- **No implementados** en esta versión. Se recomienda agregar para:
  - Versionado de datasets (DVC)
  - Registro de experimentos (MLflow)
  - Seguimiento de métricas entre versiones

## Recomendaciones para Producción

1. Reentrenar con datos reales de al menos 6 meses
2. Validar umbrales con equipo agronómico
3. Implementar monitoreo de drift (PSI) con datos de producción
4. Migrar autenticación a OAuth2/JWT
5. Implementar Redis cache para resultados de inferencia
6. Agregar logging centralizado (ELK/Datadog)
7. Configurar alertas automáticas por estado "crítico"
