"""Unit tests for the movies API endpoints."""

from collections import namedtuple
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from movie_recommender.dependencies.feed_manager import get_feed_manager
from movie_recommender.main import app
from movie_recommender.schemas.requests.movies import MovieDetails
from tests.unit.api.conftest import AUTH_HEADERS, FAKE_USER

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

_FAKE_PREF = FakePref(
    id=1,
    user_id=1,
    min_year=2000,
    max_year=2025,
    min_rating=7.0,
    include_adult=False,
    updated_at=datetime(2024, 1, 1),
)


def _make_movie_details(movie_db_id: int = 1, title: str = "Arrival") -> MovieDetails:
    return MovieDetails(
        movie_db_id=movie_db_id,
        tmdb_id=329865,
        title=title,
        poster_url="https://img/arrival.jpg",
        release_year=2016,
        rating=7.9,
        genres=["Sci-Fi"],
        is_adult=False,
        synopsis="Linguist meets aliens.",
        runtime=116,
    )


def _override_feed_manager(fm: MagicMock):
    app.dependency_overrides[get_feed_manager] = lambda: fm


def _clear_feed_override():
    app.dependency_overrides.pop(get_feed_manager, None)


class TestFetchMoviesFeedBatch:
    @patch(
        "movie_recommender.api.v1.movies.get_user_excluded_genres",
        new_callable=AsyncMock,
    )
    @patch(
        "movie_recommender.api.v1.movies.get_user_included_genres",
        new_callable=AsyncMock,
    )
    @patch(
        "movie_recommender.api.v1.movies.get_user_preferences", new_callable=AsyncMock
    )
    @patch(
        "movie_recommender.api.v1.movies.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_batch_returns_up_to_count_movies(
        self, mock_get_user, mock_prefs, mock_inc, mock_exc, client
    ):
        mock_get_user.return_value = FAKE_USER
        mock_prefs.return_value = _FAKE_PREF
        mock_inc.return_value = []
        mock_exc.return_value = []

        fm = MagicMock()
        fm.get_next_movie = AsyncMock(
            side_effect=[
                _make_movie_details(movie_db_id=1, title="M1"),
                _make_movie_details(movie_db_id=2, title="M2"),
                _make_movie_details(movie_db_id=3, title="M3"),
                None,  # queue exhausted early
                None,
            ]
        )
        _override_feed_manager(fm)

        try:
            resp = client.get("/api/v1/movies/feed/batch?count=5", headers=AUTH_HEADERS)
        finally:
            _clear_feed_override()

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        assert [m["title"] for m in data] == ["M1", "M2", "M3"]

    @patch(
        "movie_recommender.api.v1.movies.get_user_excluded_genres",
        new_callable=AsyncMock,
    )
    @patch(
        "movie_recommender.api.v1.movies.get_user_included_genres",
        new_callable=AsyncMock,
    )
    @patch(
        "movie_recommender.api.v1.movies.get_user_preferences", new_callable=AsyncMock
    )
    @patch(
        "movie_recommender.api.v1.movies.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_batch_returns_404_when_empty(
        self, mock_get_user, mock_prefs, mock_inc, mock_exc, client
    ):
        mock_get_user.return_value = FAKE_USER
        mock_prefs.return_value = _FAKE_PREF
        mock_inc.return_value = []
        mock_exc.return_value = []

        fm = MagicMock()
        fm.get_next_movie = AsyncMock(return_value=None)
        _override_feed_manager(fm)

        try:
            resp = client.get("/api/v1/movies/feed/batch?count=3", headers=AUTH_HEADERS)
        finally:
            _clear_feed_override()

        assert resp.status_code == 404

    def test_batch_rejects_count_over_limit(self, client):
        # Must override feed_manager so dependency injection doesn't hit real
        # clients during Pydantic validation failure paths.
        fm = MagicMock()
        fm.get_next_movie = AsyncMock()
        _override_feed_manager(fm)

        try:
            resp = client.get(
                "/api/v1/movies/feed/batch?count=99", headers=AUTH_HEADERS
            )
        finally:
            _clear_feed_override()

        assert resp.status_code == 422


class TestFlushMoviesFeed:
    @patch(
        "movie_recommender.api.v1.movies.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_flush_success(self, mock_get_user, client):
        mock_get_user.return_value = FAKE_USER

        fm = MagicMock()
        fm.flush_feed = AsyncMock()
        _override_feed_manager(fm)

        try:
            resp = client.delete("/api/v1/movies/feed", headers=AUTH_HEADERS)
        finally:
            _clear_feed_override()

        assert resp.status_code == 200
        assert resp.json() == {"flushed": True}
        fm.flush_feed.assert_awaited_once_with(FAKE_USER.id)

    @patch(
        "movie_recommender.api.v1.movies.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_flush_user_not_found_returns_404(self, mock_get_user, client):
        mock_get_user.return_value = None

        fm = MagicMock()
        fm.flush_feed = AsyncMock()
        _override_feed_manager(fm)

        try:
            resp = client.delete("/api/v1/movies/feed", headers=AUTH_HEADERS)
        finally:
            _clear_feed_override()

        assert resp.status_code == 404
        fm.flush_feed.assert_not_called()
