"""Shared fixtures for API handler tests."""

from collections import namedtuple
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from movie_recommender.main import app
from movie_recommender.dependencies.database import get_db

FAKE_UID = "firebase-uid-1"

FakeRow = namedtuple("FakeRow", [
    "id", "firebase_uid", "profile_image_url", "email", "created_at", "updated_at",
])

FAKE_USER = FakeRow(
    id=1,
    firebase_uid=FAKE_UID,
    profile_image_url="https://example.com/avatar.png",
    email="test@example.com",
    created_at=datetime(2024, 1, 1),
    updated_at=datetime(2024, 1, 1),
)


@pytest.fixture
def mock_db():
    return AsyncMock()


@pytest.fixture
def client(mock_db):
    """TestClient with DB mocked and Firebase auth mocked at the SDK level."""
    app.dependency_overrides[get_db] = lambda: mock_db

    with (
        patch("movie_recommender.main.initialize_firebase"),
        patch("movie_recommender.dependencies.firebase.auth.verify_id_token") as mock_verify,
        patch("movie_recommender.dependencies.firebase.auth.get_user") as mock_get_user,
    ):
        mock_verify.return_value = {"uid": FAKE_UID, "email": "test@example.com"}
        user_record = MagicMock()
        user_record.email_verified = True
        user_record.display_name = "Test User"
        mock_get_user.return_value = user_record

        with TestClient(app) as c:
            yield c

    app.dependency_overrides.clear()


AUTH_HEADERS = {"Authorization": "Bearer fake-token"}
