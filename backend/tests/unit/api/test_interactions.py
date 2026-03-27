"""Unit tests for the interactions API endpoint."""

from collections import namedtuple
from datetime import datetime
from unittest.mock import AsyncMock, patch

from tests.unit.api.conftest import AUTH_HEADERS, FAKE_USER

FakeSwipe = namedtuple("FakeSwipe", ["id", "user_id", "movie_id", "action_type", "is_supercharged", "created_at"])
FakeMovie = namedtuple("FakeMovie", ["id", "tmdb_id", "title"])


class TestRegisterMovieInteraction:
    @patch("movie_recommender.api.v1.interactions.get_recommender")
    @patch("movie_recommender.api.v1.interactions.create_swipe", new_callable=AsyncMock)
    @patch("movie_recommender.api.v1.interactions.get_movie_by_id", new_callable=AsyncMock)
    @patch("movie_recommender.api.v1.interactions.get_user_by_firebase_uid", new_callable=AsyncMock)
    def test_like_swipe_succeeds(
        self, mock_get_user, mock_get_movie, mock_create_swipe, mock_get_recommender, client
    ):
        mock_get_user.return_value = FAKE_USER
        mock_get_movie.return_value = FakeMovie(id=1, tmdb_id=100, title="Arrival")
        mock_create_swipe.return_value = FakeSwipe(
            id=42, user_id=1, movie_id=1, action_type="like",
            is_supercharged=False, created_at=datetime.now(),
        )
        mock_get_recommender.return_value = AsyncMock()

        resp = client.post(
            "/api/v1/interactions/1/swipe",
            json={"action_type": "like"},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["interaction_id"] == 42
        assert data["registered"] is True

    @patch("movie_recommender.api.v1.interactions.get_movie_by_id", new_callable=AsyncMock)
    @patch("movie_recommender.api.v1.interactions.get_user_by_firebase_uid", new_callable=AsyncMock)
    def test_movie_not_found_returns_404(self, mock_get_user, mock_get_movie, client):
        mock_get_user.return_value = FAKE_USER
        mock_get_movie.return_value = None

        resp = client.post(
            "/api/v1/interactions/999/swipe",
            json={"action_type": "like"},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 404

    @patch("movie_recommender.api.v1.interactions.get_user_by_firebase_uid", new_callable=AsyncMock)
    def test_user_not_found_returns_404(self, mock_get_user, client):
        mock_get_user.return_value = None

        resp = client.post(
            "/api/v1/interactions/1/swipe",
            json={"action_type": "like"},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 404

    def test_supercharged_skip_returns_400(self, client):
        resp = client.post(
            "/api/v1/interactions/1/swipe",
            json={"action_type": "skip", "is_supercharged": True},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 400
