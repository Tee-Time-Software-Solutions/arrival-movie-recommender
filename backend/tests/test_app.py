"""Smoke tests for the FastAPI app (OpenAPI, root)."""
from fastapi.testclient import TestClient


def test_openapi_docs_available(client: TestClient) -> None:
    """GET /docs returns 200 so CI can confirm app serves."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_json_available(client: TestClient) -> None:
    """GET /openapi.json returns 200 and valid OpenAPI structure."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert data.get("info", {}).get("title") == "Movie Recommender"
