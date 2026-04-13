"""Unit tests for the watchlist API endpoints."""

from collections import namedtuple
from unittest.mock import AsyncMock, patch

from movie_recommender.schemas.requests.movies import MovieDetails
from tests.unit.api.conftest import AUTH_HEADERS, FAKE_USER

FakeMovie = namedtuple("FakeMovie", ["id", "tmdb_id", "title"])
_FAKE_MOVIE = FakeMovie(id=1, tmdb_id=329865, title="Arrival")


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


class TestAddMovieToWatchlist:
    @patch(
        "movie_recommender.api.v1.watchlist.add_to_watchlist", new_callable=AsyncMock
    )
    @patch(
        "movie_recommender.api.v1.watchlist.get_movie_by_id", new_callable=AsyncMock
    )
    @patch(
        "movie_recommender.api.v1.watchlist.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_add_success(
        self, mock_get_user, mock_get_movie, mock_add, client
    ):
        mock_get_user.return_value = FAKE_USER
        mock_get_movie.return_value = _FAKE_MOVIE
        mock_add.return_value = 1

        resp = client.post("/api/v1/watchlist/1", headers=AUTH_HEADERS)

        assert resp.status_code == 200
        assert resp.json() == {"movie_id": 1, "added": True}
        mock_add.assert_awaited_once()

    @patch(
        "movie_recommender.api.v1.watchlist.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_add_user_not_found(self, mock_get_user, client):
        mock_get_user.return_value = None

        resp = client.post("/api/v1/watchlist/1", headers=AUTH_HEADERS)
        assert resp.status_code == 404

    @patch(
        "movie_recommender.api.v1.watchlist.get_movie_by_id", new_callable=AsyncMock
    )
    @patch(
        "movie_recommender.api.v1.watchlist.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_add_movie_not_found(self, mock_get_user, mock_get_movie, client):
        mock_get_user.return_value = FAKE_USER
        mock_get_movie.return_value = None

        resp = client.post("/api/v1/watchlist/999", headers=AUTH_HEADERS)
        assert resp.status_code == 404

    @patch(
        "movie_recommender.api.v1.watchlist.add_to_watchlist", new_callable=AsyncMock
    )
    @patch(
        "movie_recommender.api.v1.watchlist.get_movie_by_id", new_callable=AsyncMock
    )
    @patch(
        "movie_recommender.api.v1.watchlist.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_add_duplicate_returns_409(
        self, mock_get_user, mock_get_movie, mock_add, client
    ):
        mock_get_user.return_value = FAKE_USER
        mock_get_movie.return_value = _FAKE_MOVIE
        mock_add.return_value = None  # indicates already in watchlist

        resp = client.post("/api/v1/watchlist/1", headers=AUTH_HEADERS)
        assert resp.status_code == 409


class TestRemoveMovieFromWatchlist:
    @patch(
        "movie_recommender.api.v1.watchlist.remove_from_watchlist",
        new_callable=AsyncMock,
    )
    @patch(
        "movie_recommender.api.v1.watchlist.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_remove_success(self, mock_get_user, mock_remove, client):
        mock_get_user.return_value = FAKE_USER
        mock_remove.return_value = True

        resp = client.delete("/api/v1/watchlist/1", headers=AUTH_HEADERS)

        assert resp.status_code == 200
        assert resp.json() == {"movie_id": 1, "removed": True}
        mock_remove.assert_awaited_once()

    @patch(
        "movie_recommender.api.v1.watchlist.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_remove_user_not_found(self, mock_get_user, client):
        mock_get_user.return_value = None

        resp = client.delete("/api/v1/watchlist/1", headers=AUTH_HEADERS)
        assert resp.status_code == 404

    @patch(
        "movie_recommender.api.v1.watchlist.remove_from_watchlist",
        new_callable=AsyncMock,
    )
    @patch(
        "movie_recommender.api.v1.watchlist.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_remove_not_in_watchlist_returns_404(
        self, mock_get_user, mock_remove, client
    ):
        mock_get_user.return_value = FAKE_USER
        mock_remove.return_value = False  # nothing to remove

        resp = client.delete("/api/v1/watchlist/999", headers=AUTH_HEADERS)
        assert resp.status_code == 404


class TestGetWatchlistMovies:
    @patch(
        "movie_recommender.api.v1.watchlist.movies_to_details_bulk",
        new_callable=AsyncMock,
    )
    @patch(
        "movie_recommender.api.v1.watchlist.get_user_watchlist",
        new_callable=AsyncMock,
    )
    @patch(
        "movie_recommender.api.v1.watchlist.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_list_returns_paginated_movies(
        self, mock_get_user, mock_list, mock_bulk, client
    ):
        mock_get_user.return_value = FAKE_USER
        mock_list.return_value = ([1, 2], 2)
        mock_bulk.return_value = [
            _make_movie_details(movie_db_id=1, title="Arrival"),
            _make_movie_details(movie_db_id=2, title="Interstellar"),
        ]

        resp = client.get(
            "/api/v1/watchlist?limit=20&offset=0", headers=AUTH_HEADERS
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["limit"] == 20
        assert data["offset"] == 0
        assert len(data["items"]) == 2
        assert data["items"][0]["title"] == "Arrival"
        # Verify the ownership scope: CRUD was called with the user's id
        mock_list.assert_awaited_once()
        called_args = mock_list.await_args.args
        assert called_args[1] == FAKE_USER.id

    @patch(
        "movie_recommender.api.v1.watchlist.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_list_user_not_found(self, mock_get_user, client):
        mock_get_user.return_value = None

        resp = client.get("/api/v1/watchlist", headers=AUTH_HEADERS)
        assert resp.status_code == 404

    @patch(
        "movie_recommender.api.v1.watchlist.movies_to_details_bulk",
        new_callable=AsyncMock,
    )
    @patch(
        "movie_recommender.api.v1.watchlist.get_user_watchlist",
        new_callable=AsyncMock,
    )
    @patch(
        "movie_recommender.api.v1.watchlist.get_user_by_firebase_uid",
        new_callable=AsyncMock,
    )
    def test_list_empty_watchlist(
        self, mock_get_user, mock_list, mock_bulk, client
    ):
        mock_get_user.return_value = FAKE_USER
        mock_list.return_value = ([], 0)
        mock_bulk.return_value = []

        resp = client.get("/api/v1/watchlist", headers=AUTH_HEADERS)

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []
