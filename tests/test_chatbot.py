"""
SMISIA — Tests del Chatbot
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.chatbot.bot import format_chat_response, DIALOG_EXAMPLES  # noqa: E402


class TestChatbot:
    def test_status_with_data(self):
        """Debe generar respuesta de estado con datos."""
        cached = {
            "status": "problema",
            "confidence": 0.78,
            "summary": "Humedad en aumento.",
            "metrics": {
                "temperature_c": {"value": 28.4, "unit": "°C"},
                "humidity_pct": {"value": 16.8, "unit": "%"},
            },
            "trend": {"humidity_pct_delta_24h": 1.8},
            "explanations": [
                {"feature": "humidity_delta_72h", "impact": 0.42},
            ],
            "recommended_action": "Inspección física.",
            "raw_scores": {
                "bien": 0.05,
                "tolerable": 0.12,
                "problema": 0.78,
                "critico": 0.05,
            },
        }
        response = format_chat_response(
            "¿Cuál es el estado de la silobolsa A12?",
            silo_id="A12",
            cached_data=cached,
        )
        assert "PROBLEMA" in response.brief
        assert "A12" in response.brief
        assert response.detail is not None

    def test_status_no_data(self):
        """Sin datos, debe indicar que no hay info."""
        response = format_chat_response(
            "¿Cuál es el estado del silo X99?",
            silo_id="X99",
            cached_data=None,
        )
        assert "No tengo datos" in response.brief

    def test_prediction_query(self):
        """Consulta de predicción."""
        cached = {
            "status": "problema",
            "confidence": 0.78,
            "raw_scores": {
                "bien": 0.05,
                "tolerable": 0.12,
                "problema": 0.78,
                "critico": 0.05,
            },
            "explanations": [
                {"feature": "humidity_slope", "impact": 0.5},
            ],
        }
        response = format_chat_response(
            "¿Va a empeorar en 3 días?",
            silo_id="A12",
            cached_data=cached,
        )
        assert "probabilidad" in response.brief.lower()
        assert "3" in response.brief

    def test_unknown_intent(self):
        """Intención desconocida sin silo."""
        response = format_chat_response("hola mundo")
        assert "No entendí" in response.brief

    def test_dialog_examples_exist(self):
        """Los ejemplos de diálogo deben existir."""
        assert len(DIALOG_EXAMPLES) >= 3
        for ex in DIALOG_EXAMPLES:
            assert "user" in ex
            assert "bot" in ex
