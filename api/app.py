"""
AGRILION — FastAPI Application
=================================

Punto de entrada de la API REST.
Configura CORS, middleware, y registra todos los routers.

Para ejecutar:
    uvicorn api.app:app --reload --port 8000
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Asegurar imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src import __version__
from src.config import API_CONFIG
from api.routes import router, initialize_model

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicialización al arrancar la API."""
    logging.info("🚀 Iniciando AGRILION API...")
    initialize_model()
    logging.info("✅ AGRILION API lista")
    yield
    logging.info("🛑 AGRILION API detenida")


# Crear app
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar router
app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "name": "AGRILION API",
        "description": "Sistema ML de análisis de silobolsas agrícolas",
        "version": __version__,
        "docs": "/docs",
        "endpoints": {
            "health": "/api/v1/health",
            "predict": "/api/v1/predict",
            "analyze": "/api/v1/analyze",
            "model_status": "/api/v1/model/status",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=True,
    )
