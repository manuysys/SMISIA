# ========================================================
# SMISIA — Dockerfile
# ========================================================
FROM python:3.11-slim

WORKDIR /app

# Instalar deps del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ src/
COPY config.yml .
COPY serve.py .
COPY train.py .
COPY eval.py .
COPY generate_dataset.py .

# Copiar modelos si existen
COPY models/ models/

# Copiar datos si existen
COPY data/ data/

# Puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/metrics')" || exit 1

# Arrancar
CMD ["python", "serve.py"]
