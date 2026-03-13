import shap
import xgboost as xgb
import numpy as np
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_shap")

try:
    # Cargar el modelo guardado
    model_path = "models/xgboost_model.joblib"
    model = joblib.load(model_path)
    logger.info(f"Modelo cargado: {type(model)}")
    
    # Datos dummy
    X = np.random.rand(100, 198).astype(np.float32)
    
    logger.info("Iniciando TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    logger.info("Calculando shap_values...")
    # Intento 1: API vieja
    try:
        sv = explainer.shap_values(X)
        logger.info(f"API Vieja exitosa! Tipo: {type(sv)}")
        if isinstance(sv, list):
            logger.info(f"Lista de largo {len(sv)}")
        else:
            logger.info(f"Array de shape {sv.shape}")
    except Exception as e:
        logger.error(f"API Vieja falló: {e}")
        
    # Intento 2: API nueva
    try:
        sv2 = explainer(X)
        logger.info(f"API Nueva exitosa! Tipo: {type(sv2)}")
        logger.info(f"Values shape: {sv2.values.shape}")
    except Exception as e:
        logger.error(f"API Nueva falló: {e}")

except Exception as e:
    logger.error(f"Error general: {e}")
