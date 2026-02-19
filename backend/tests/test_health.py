"""Unit tests for health API (proves CI can execute tests)."""
import pytest
from fastapi.testclient import TestClient


def test_health_ping_returns_200(client: TestClient) -> None:
    """GET /api/v1/health/ping returns 200 and pong."""
    response = client.get("/api/v1/health/ping")
    assert response.status_code == 200
    assert response.json() == {"response": "pong"}


def test_health_ping_response_shape(client: TestClient) -> None:
    """Health ping response has expected keys."""
    response = client.get("/api/v1/health/ping")
    data = response.json()
    assert "response" in data
    assert data["response"] == "pong"
