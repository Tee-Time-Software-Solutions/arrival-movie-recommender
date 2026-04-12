"""Unit tests for the interactions API endpoint."""

from collections import namedtuple
from unittest.mock import AsyncMock, patch

from tests.unit.api.conftest import AUTH_HEADERS, FAKE_USER

FakeMovie = namedtuple("FakeMovie", ["id", "tmdb_id", "title"])


class TestRegisterMovieInteraction:
    @patch(
        "movie_recommender.api.v1.interactions.update_beacon_on_swipe",
        new_callable=AsyncMock,
    )
    @patch(
        "movie_recommender.api.v1.interactions.persist_swipe", new_callable=AsyncMock
    )
    @patch(
        "movie_recommender.api.v1.interactions.get_movie_by_id", new_callable=AsyncMock
    )
    @patch(
        "movie_recommender.api.v1.interactions.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_like_swipe_succeeds(
        self,
        mock_get_user,
        mock_get_movie,
        mock_enqueue,
        mock_beacon,
        client,
    ):
        mock_get_user.return_value = FAKE_USER
        mock_get_movie.return_value = FakeMovie(id=1, tmdb_id=100, title="Arrival")

        from movie_recommender.dependencies.recommender import get_recommender
        from movie_recommender.dependencies.redis import get_async_redis
        from movie_recommender.dependencies.neo4j import get_neo4j_driver
        from movie_recommender.main import app

        app.dependency_overrides[get_async_redis] = lambda: AsyncMock()
        app.dependency_overrides[get_neo4j_driver] = lambda: AsyncMock()
        app.dependency_overrides[get_recommender] = lambda: AsyncMock()

        resp = client.post(
            "/api/v1/interactions/1/swipe",
            json={"action_type": "like"},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["interaction_id"] == 0
        assert data["registered"] is True

    @patch(
        "movie_recommender.api.v1.interactions.get_movie_by_id", new_callable=AsyncMock
    )
    @patch(
        "movie_recommender.api.v1.interactions.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_movie_not_found_returns_404(self, mock_get_user, mock_get_movie, client):
        mock_get_user.return_value = FAKE_USER
        mock_get_movie.return_value = None

        from movie_recommender.dependencies.redis import get_async_redis
        from movie_recommender.dependencies.neo4j import get_neo4j_driver
        from movie_recommender.main import app

        app.dependency_overrides[get_async_redis] = lambda: AsyncMock()
        app.dependency_overrides[get_neo4j_driver] = lambda: AsyncMock()

        resp = client.post(
            "/api/v1/interactions/999/swipe",
            json={"action_type": "like"},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 404

    @patch(
        "movie_recommender.api.v1.interactions.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_user_not_found_returns_404(self, mock_get_user, client):
        mock_get_user.return_value = None

        from movie_recommender.dependencies.redis import get_async_redis
        from movie_recommender.dependencies.neo4j import get_neo4j_driver
        from movie_recommender.main import app

        app.dependency_overrides[get_async_redis] = lambda: AsyncMock()
        app.dependency_overrides[get_neo4j_driver] = lambda: AsyncMock()

        resp = client.post(
            "/api/v1/interactions/1/swipe",
            json={"action_type": "like"},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 404

    def test_supercharged_skip_returns_400(self, client):
        from movie_recommender.dependencies.redis import get_async_redis
        from movie_recommender.dependencies.neo4j import get_neo4j_driver
        from movie_recommender.main import app

        app.dependency_overrides[get_async_redis] = lambda: AsyncMock()
        app.dependency_overrides[get_neo4j_driver] = lambda: AsyncMock()

        resp = client.post(
            "/api/v1/interactions/1/swipe",
            json={"action_type": "skip", "is_supercharged": True},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 400
