"""
SMISIA — Aplicación FastAPI
Punto de entrada de la API REST.
"""
import logging
import os
import sys

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Asegurar que el proyecto está en el path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.api.routes import router, init_state  # noqa: E402
from src.api.security import APIKeyMiddleware  # noqa: E402

logger = logging.getLogger("smisia.api")


def create_app(config_path: str = None) -> FastAPI:
    """Crea y configura la aplicación FastAPI."""
    config = load_config(config_path)
    api_cfg = config.get("api", {})

    app = FastAPI(
        title="SMISIA — API de Monitoreo de Silobolsas",
        description=(
            "Sistema de Monitoreo e IA para Silobolsas. "
            "Clasificación, predicción, detección de anomalías y explicabilidad."
        ),
        version=config["project"]["version"],
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_cfg.get("cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API Key
    app.add_middleware(APIKeyMiddleware, api_key=api_cfg.get("api_key"))

    # Cargar modelos al iniciar
    @app.on_event("startup")
    async def load_models():
        models_dir = config["paths"]["models_dir"]
        state = {}

        try:
            model_data = {}
            model_path = os.path.join(models_dir, "xgboost_model.joblib")
            if os.path.exists(model_path):
                state["xgb_model"] = joblib.load(model_path)
                logger.info("✓ XGBoost model cargado")

            feat_path = os.path.join(models_dir, "feature_columns.joblib")
            if os.path.exists(feat_path):
                state["feature_columns"] = joblib.load(feat_path)
                logger.info(f"✓ {len(state['feature_columns'])} feature columns cargadas")

            imp_path = os.path.join(models_dir, "feature_importance.joblib")
            if os.path.exists(imp_path):
                state["feature_importance"] = joblib.load(imp_path)

            boot_path = os.path.join(models_dir, "bootstrap_models.joblib")
            if os.path.exists(boot_path):
                state["bootstrap_models"] = joblib.load(boot_path)
                logger.info(f"✓ {len(state['bootstrap_models'])} bootstrap models cargados")

            anom_path = os.path.join(models_dir, "anomaly_model.joblib")
            if os.path.exists(anom_path):
                state["anomaly_model"] = joblib.load(anom_path)
                state["anomaly_scaler"] = joblib.load(
                    os.path.join(models_dir, "anomaly_scaler.joblib")
                )
                logger.info("✓ Anomaly detector cargado")

        except Exception as e:
            logger.error(f"Error cargando modelos: {e}")

        init_state(state)
        logger.info("API lista para inferencia")

    # Rutas
    app.include_router(router)

    return app


# Instancia global para uvicorn
app = create_app()
