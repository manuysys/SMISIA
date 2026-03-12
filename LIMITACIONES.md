# SMISIA — Limitaciones y Supuestos del Modelo

Este documento detalla las fronteras operativas y supuestos técnicos de la IA de SMISIA (v1.0.0).

## 1. Supuestos del Dataset (Sim-to-Real)
- **Datos Sintéticos**: El modelo actual ha sido entrenado con datos sintéticos realistas. Puede haber un "gap" de rendimiento cuando se enfrenten datos reales de sensores físicos debido a ruidos no modelados.
- **Frecuencia de Muestreo**: Se asume una lectura cada 2 horas. Cambios significativos en la frecuencia pueden afectar el cálculo de pendientes (slopes).

## 2. Limitaciones del Modelo (XGBoost Ensemble)
- **Umbral de Incertidumbre**: El modelo utiliza un ensemble de 5 modelos. Si la desviación estándar entre modelos es > 0.12, la predicción debe tomarse con cautela y requiere revisión humana.
- **Variables Críticas**: El modelo depende fuertemente de `co2_ppm` y `humidity_pct`. La falla de cualquiera de estos sensores degrada significativamente la precisión.

## 3. Alertas y Falsos Positivos
- **Deriva de Sensores**: Las derivas técnicas (drift) pueden ser confundidas con procesos de deterioro si no se monitorea la salud del sensor (`sensor_health`).
- **Casos de Borde**: Eventos climáticos extremos (ej. granizo) que afecten la presión externa no están modelados directamente.

## 4. Mantenimiento Requerido
- **PSI (Population Stability Index)**: Si el PSI acumulado mensual supera 0.25, es mandatario realizar un re-entrenamiento (Retrain) con los nuevos datos recolectados.
- **Etiquetado Activo**: El sistema depende de la retroalimentación humana en casos de alta entropía para refinar su capacidad de detección.
