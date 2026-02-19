"""Smoke tests for the FastAPI app (docs, root)."""
from fastapi.testclient import TestClient


def test_openapi_docs_available(client: TestClient) -> None:
    """GET /docs returns 200 so CI can confirm app serves."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_root_available(client: TestClient) -> None:
    """GET / returns 404 or 200; app is mounted and responding."""
    response = client.get("/")
    assert response.status_code in (200, 404, 307)
