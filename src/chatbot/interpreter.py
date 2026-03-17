"""
SMISIA — Intérprete de Intenciones del Chatbot
Usa similaridad para detectar qué quiere el usuario.
"""

import re
import numpy as np

# Definición de intenciones y frases de ejemplo (clústeres)
INTENTS = {
    "status": [
        "estado del silo",
        "cómo está la silobolsa",
        "dame el status",
        "condición actual",
        "está todo bien",
        "qué onda con el silo",
        "reporte de situación",
        "estátus",
        "estado"
    ],
    "prediction": [
        "va a empeorar",
        "riesgo futuro",
        "predicción a tres días",
        "cómo va a estar mañana",
        "pronóstico",
        "se va a pudrir",
        "cuándo tengo que sacar el grano",
        "probabilidad de problema",
        "futuro",
        "mañana",
        "prediccion"
    ],
    "trend": [
        "histórico de humedad",
        "gráfica de tendencia",
        "cómo varió la temperatura",
        "evolución de los últimos días",
        "comportamiento semanal",
        "tendencia",
        "evolucion",
        "grafico"
    ],
    "global_status": [
        "estado general",
        "salud de los silos",
        "cómo están todos",
        "situación global",
        "estado de la planta",
        "pasando algo raro",
        "inusual",
        "global"
    ],
    "ranking": [
        "peor",
        "más crítico",
        "mas critico",
        "top",
        "ranking",
        "riesgo extremo",
        "mayores problemas"
    ],
    "speed": [
        "deteriorando más rápido",
        "peor más rápido",
        "deterioro",
        "más rápido",
        "empeorando"
    ],
    "sensor_health": [
        "sensores andando bien",
        "sensores fallando",
        "problemas de hardware",
        "sensor",
        "falla técnica",
        "malfuncionamiento"
    ]
}

def clean_text(text: str) -> str:
    """Limpieza básica para comparación."""
    text = text.lower().strip()
    # Quitar tildes básicas
    text = text.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    # Quitar signos de interrogación y puntuación
    text = re.sub(r'[^\w\s]', '', text)
    return text

def get_intent(message: str) -> str:
    """
    Detecta la intención del mensaje.
    Fallback: Si no hay match claro, busca palabras clave.
    """
    clean_msg = clean_text(message)
    
    # 1. Búsqueda exacta en clústeres
    for intent, examples in INTENTS.items():
        for ex in examples:
            if clean_text(ex) in clean_msg:
                return intent
                
    # 2. Búsqueda de palabras clave críticas
    keywords = {
        "status": ["estado", "como", "bien", "mal", "situacion", "status"],
        "prediction": ["empeorar", "futuro", "pronostico", "mañana", "despues", "prediccion"],
        "trend": ["tendencia", "grafico", "evolucion", "historia", "pasado"]
    }
    
    # Contar matches por intención
    scores = {intent: 0 for intent in keywords}
    for intent, kws in keywords.items():
        for kw in kws:
            if kw in clean_msg:
                scores[intent] += 1
                
    best_intent = max(scores, key=scores.get)
    if scores[best_intent] > 0:
        return best_intent
        
    return "unknown"

def extract_silo_id(message: str) -> str:
    """Intenta extraer un ID de silo (e.g. A12, SILO_001, B5) del mensaje."""
    # Patrones comunes: A12, SILO_001, B5, C102. Permitimos 1 a 4 dígitos para mayor flexibilidad.
    match = re.search(r'(?:silo(?:bolsa)?\s+)?([a-zA-Z]\d{1,4}|silo_\d{3})', message, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None
