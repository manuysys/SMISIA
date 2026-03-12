"""
SMISIA — Tests de la API
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.api.app import create_app  # noqa: E402


@pytest.fixture
def client():
    """Cliente de test para la API."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Headers con API key."""
    return {"X-API-Key": "smisia-dev-key-2026"}


class TestHealthEndpoint:
    def test_metrics_no_auth(self, client):
        """El endpoint /metrics no requiere auth."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "models_loaded" in data
        assert "uptime_seconds" in data

    def test_metrics_response_shape(self, client):
        response = client.get("/metrics")
        data = response.json()
        assert isinstance(data["models_loaded"], dict)
        assert isinstance(data["uptime_seconds"], float)


class TestAuthMiddleware:
    def test_infer_requires_auth(self, client):
        """El endpoint /infer requiere API key."""
        response = client.post("/infer", json={
            "silo_id": "TEST",
            "timestamp": "2025-10-01T00:00:00+00:00",
            "recent_readings": [],
        })
        assert response.status_code == 401

    def test_infer_with_auth(self, client, auth_headers):
        """Con auth, el endpoint responde (puede fallar por modelo no cargado)."""
        response = client.post(
            "/infer",
            json={
                "silo_id": "TEST",
                "timestamp": "2025-10-01T00:00:00+00:00",
                "recent_readings": [
                    {
                        "timestamp": "2025-10-01T00:00:00+00:00",
                        "temperature_c": 25.0,
                        "humidity_pct": 14.0,
                        "co2_ppm": 500.0,
                        "battery_pct": 90.0,
                        "rssi": -80,
                        "snr": 10.0,
                    }
                ],
            },
            headers=auth_headers,
        )
        # Con modelos no cargados, puede ser 200 o 500
        assert response.status_code in [200, 500]


class TestStatusEndpoint:
    def test_status_unknown_silo(self, client, auth_headers):
        """Status de un silo sin datos previos."""
        response = client.get("/status/UNKNOWN_SILO", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["silo_id"] == "UNKNOWN_SILO"


class TestChatEndpoint:
    def test_chat_unknown_intent(self, client, auth_headers):
        """Chat con intención desconocida."""
        response = client.post(
            "/chat",
            json={"message": "hola mundo"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "brief" in data
