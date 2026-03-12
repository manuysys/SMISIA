#!/bin/bash
# ========================================================
# SMISIA — Script de Despliegue
# ========================================================
set -e

echo "=========================================="
echo "  SMISIA — Deploy"
echo "=========================================="

# 1. Verificar que Docker está disponible
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker no está instalado"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "ERROR: Docker Compose no está instalado"
    exit 1
fi

# 2. Crear directorios necesarios
mkdir -p data models

# 3. Generar dataset si no existe
if [ ! -f "data/synthetic_silo_dataset.csv" ]; then
    echo "[1/4] Generando dataset sintético..."
    pip install -r requirements.txt -q 2>/dev/null || true
    python generate_dataset.py
else
    echo "[1/4] Dataset ya existe, omitiendo generación."
fi

# 4. Entrenar modelos si no existen
if [ ! -f "models/xgboost_model.joblib" ]; then
    echo "[2/4] Entrenando modelos..."
    python train.py --skip-lstm
else
    echo "[2/4] Modelos ya existen, omitiendo entrenamiento."
fi

# 5. Construir imagen Docker
echo "[3/4] Construyendo imagen Docker..."
docker build -t smisia-api:latest .

# 6. Levantar servicios
echo "[4/4] Levantando servicios..."
docker compose up -d

echo ""
echo "=========================================="
echo "  SMISIA desplegado exitosamente!"
echo "=========================================="
echo ""
echo "  API:     http://localhost:8000"
echo "  Docs:    http://localhost:8000/docs"
echo "  Health:  http://localhost:8000/metrics"
echo ""
echo "  Usar API key: smisia-dev-key-2026"
echo "  Header:  X-API-Key: smisia-dev-key-2026"
echo ""
