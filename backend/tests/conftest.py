"""Shared pytest fixtures for FastAPI TestClient (FastAPI-standard)."""
import pytest
from fastapi.testclient import TestClient

from movie_recommender.main import app


@pytest.fixture
def client() -> TestClient:
    """Yield a TestClient bound to the app for the duration of the test."""
    with TestClient(app) as c:
        yield c
