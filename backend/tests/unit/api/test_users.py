"""Unit tests for the users API endpoints."""

from collections import namedtuple
from datetime import datetime
from unittest.mock import AsyncMock, patch

from tests.unit.api.conftest import AUTH_HEADERS, FAKE_UID, FAKE_USER

FakePref = namedtuple(
    "FakePref",
    [
        "id",
        "user_id",
        "min_year",
        "max_year",
        "min_rating",
        "include_adult",
        "updated_at",
    ],
)


class TestRegisterUser:
    @patch("movie_recommender.api.v1.users.create_user", new_callable=AsyncMock)
    @patch(
        "movie_recommender.api.v1.users.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_register_new_user(self, mock_get_user, mock_create_user, client):
        mock_get_user.return_value = None
        mock_create_user.return_value = FAKE_USER

        resp = client.post(
            "/api/v1/users/register",
            json={
                "firebase_uid": FAKE_UID,
                "profile_image_url": "https://example.com/avatar.png",
                "email": "test@example.com",
            },
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 200
        assert resp.json()["id"] == 1
        assert resp.json()["email"] == "test@example.com"

    @patch(
        "movie_recommender.api.v1.users.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_duplicate_user_returns_409(self, mock_get_user, client):
        mock_get_user.return_value = FAKE_USER

        resp = client.post(
            "/api/v1/users/register",
            json={
                "firebase_uid": FAKE_UID,
                "profile_image_url": "https://example.com/avatar.png",
                "email": "test@example.com",
            },
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 409


class TestGetProfileSummary:
    @patch(
        "movie_recommender.api.v1.users.get_user_excluded_genres",
        new_callable=AsyncMock,
    )
    @patch(
        "movie_recommender.api.v1.users.get_user_included_genres",
        new_callable=AsyncMock,
    )
    @patch(
        "movie_recommender.api.v1.users.get_user_preferences", new_callable=AsyncMock
    )
    @patch("movie_recommender.api.v1.users.get_user_analytics", new_callable=AsyncMock)
    @patch(
        "movie_recommender.api.v1.users.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_returns_full_profile(
        self, mock_get_user, mock_analytics, mock_prefs, mock_inc, mock_exc, client
    ):
        mock_get_user.return_value = FAKE_USER
        mock_analytics.return_value = {
            "total_swipes": 10,
            "total_likes": 6,
            "total_dislikes": 4,
            "total_seen": 10,
            "top_genres": ["Drama"],
        }
        mock_prefs.return_value = FakePref(
            id=1,
            user_id=1,
            min_year=2000,
            max_year=2025,
            min_rating=7.0,
            include_adult=False,
            updated_at=datetime.now(),
        )
        mock_inc.return_value = ["Sci-Fi"]
        mock_exc.return_value = ["Horror"]

        resp = client.get(f"/api/v1/users/{FAKE_UID}/summary", headers=AUTH_HEADERS)

        assert resp.status_code == 200
        data = resp.json()
        assert data["stats"]["total_likes"] == 6
        assert data["preferences"]["included_genres"] == ["Sci-Fi"]
        assert data["preferences"]["excluded_genres"] == ["Horror"]
        assert data["preferences"]["max_release_year"] == 2025

    @patch(
        "movie_recommender.api.v1.users.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_user_not_found_returns_404(self, mock_get_user, client):
        mock_get_user.return_value = None

        resp = client.get(f"/api/v1/users/{FAKE_UID}/summary", headers=AUTH_HEADERS)
        assert resp.status_code == 404


class TestUpdatePreferences:
    @patch(
        "movie_recommender.api.v1.users.update_user_preferences", new_callable=AsyncMock
    )
    @patch(
        "movie_recommender.api.v1.users.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_update_preferences(self, mock_get_user, mock_update, client):
        mock_get_user.return_value = FAKE_USER

        resp = client.patch(
            f"/api/v1/users/{FAKE_UID}/preferences",
            json={
                "included_genres": ["Sci-Fi", "Drama"],
                "excluded_genres": ["Horror"],
                "min_release_year": 2000,
                "max_release_year": 2025,
                "min_rating": 7.0,
                "include_adult": False,
            },
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["included_genres"] == ["Sci-Fi", "Drama"]
        assert data["max_release_year"] == 2025
        mock_update.assert_called_once()

    @patch(
        "movie_recommender.api.v1.users.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_user_not_found_returns_404(self, mock_get_user, client):
        mock_get_user.return_value = None

        resp = client.patch(
            f"/api/v1/users/{FAKE_UID}/preferences",
            json={"include_adult": True},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 404
