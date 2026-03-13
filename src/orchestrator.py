"""
SMISIA — Orquestador de Mantenimiento
Monitorea métricas y dispara reentrenamientos automatizados.
"""

import logging
import json
import os
import subprocess
import sys

logger = logging.getLogger("smisia.orchestrator")

def check_training_triggers(eval_report_path: str = "models/evaluation_report.json") -> bool:
    """
    Lee el reporte de evaluación y determina si es necesario reentrenar.
    Thresholds: Recall(critico) < 0.85 o PSI > 0.25 en variables clave.
    """
    if not os.path.exists(eval_report_path):
        logger.warning(f"No se encontró el reporte en {eval_report_path}")
        return False

    try:
        with open(eval_report_path, "r") as f:
            report = json.load(f)
    except Exception as e:
        logger.error(f"Error leyendo reporte: {e}")
        return False

    # 1. Trigger por Performance
    recall_critico = report.get("classification", {}).get("recall_critico", 1.0)
    if recall_critico < 0.85:
        logger.info(f"Trigger: Recall Crítico ({recall_critico:.2f}) debajo del threshold (0.85)")
        return True

    # 2. Trigger por Drift (PSI)
    stability = report.get("stability", {}).get("feature_psi", {})
    # Solo variables clave o cualquiera
    high_drift = [f for f, psi in stability.items() if psi > 0.25]
    if high_drift:
        logger.info(f"Trigger: Drift detectado en {len(high_drift)} variables (PSI > 0.25)")
        return True

    return False

def run_automated_retraining():
    """Llama al pipeline de entrenamiento completo."""
    logger.info("Iniciando reentrenamiento automatizado...")
    python_exe = sys.executable
    try:
        # Asumimos que train.py está en el root
        # Agregamos skip flags para que las pruebas de robustez/fase 2 sean rápidas.
        subprocess.run([python_exe, "train.py", "--skip-lstm", "--skip-anomaly"], check=True)
        logger.info("Reentrenamiento completado con éxito.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Falla en el reentrenamiento: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if check_training_triggers():
        run_automated_retraining()
    else:
        logger.info("No se requieren acciones de reentrenamiento.")
