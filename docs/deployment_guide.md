# SMISIA — Guía de Despliegue

## Despliegue Local (Desarrollo)

### Requisitos
- Python 3.11+
- pip

### Pasos
```bash
pip install -r requirements.txt
python generate_dataset.py
python train.py
python serve.py
```

## Despliegue con Docker

### Requisitos
- Docker 24+
- Docker Compose v2

### Despliegue automático
```bash
bash deploy.sh
```

### Despliegue manual
```bash
# Crear directorios
mkdir -p data models

# Generar datos y entrenar (si es necesario)
pip install -r requirements.txt
python generate_dataset.py
python train.py --skip-lstm

# Construir y levantar
docker-compose up -d --build
```

### Servicios
| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| api | 8000 | FastAPI REST API |
| redis | 6379 | Cache (Redis) |
| postgres | 5432 | Metadata (PostgreSQL) |

### Verificar
```bash
# Health check
curl http://localhost:8000/metrics

# Logs
docker-compose logs api

# Parar
docker-compose down
```

## Variables de Entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| SMISIA_CONFIG | config.yml | Ruta al archivo de configuración |
| SMISIA_API_KEY | smisia-dev-key-2026 | API key para autenticación |

## Escalabilidad

La API es stateless y puede escalarse horizontalmente:
```bash
docker-compose up -d --scale api=3
```

Agregar un load balancer (nginx/traefik) frente a las instancias.
