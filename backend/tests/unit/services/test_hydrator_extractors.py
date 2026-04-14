"""Unit tests for TMDBFetcher's pure parsing / extractor helpers and MovieHydrator.get_or_fetch_movie."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from movie_recommender.services.recommender.pipeline.hydrator.main import (
    MovieHydrator,
    TMDBFetcher,
    _parse_ml_title,
)


def _make_fetcher() -> TMDBFetcher:
    with patch.object(TMDBFetcher, "__init__", lambda self: None):
        fetcher = TMDBFetcher.__new__(TMDBFetcher)
        fetcher.IMG_URL = "https://img/"
        fetcher.BASE_URL = "https://api.tmdb"
        fetcher.TMDB_API_KEY = "fake"
        return fetcher


class TestParseMlTitle:
    def test_strips_year_and_moves_article(self):
        assert _parse_ml_title("Matrix, The (1999)") == ("The Matrix", 1999)

    def test_handles_aka_suffix(self):
        clean, year = _parse_ml_title(
            "Amélie (a.k.a. Fabuleux destin d'Amélie Poulain, Le) (2001)"
        )
        assert year == 2001
        assert "a.k.a" not in clean

    def test_no_year(self):
        clean, year = _parse_ml_title("Arrival")
        assert clean == "Arrival"
        assert year is None


class TestExtractors:
    def test_extract_trailer_url_picks_youtube_trailer(self):
        fetcher = _make_fetcher()
        detail = {
            "videos": {
                "results": [
                    {"type": "Teaser", "site": "YouTube", "key": "aaa"},
                    {"type": "Trailer", "site": "YouTube", "key": "bbb"},
                ]
            }
        }
        assert (
            fetcher._extract_trailer_url(detail)
            == "https://www.youtube.com/watch?v=bbb"
        )

    def test_extract_trailer_url_returns_none_when_missing(self):
        assert _make_fetcher()._extract_trailer_url({}) is None

    def test_extract_providers_flatrate_rent_buy(self):
        detail = {
            "watch/providers": {
                "results": {
                    "US": {
                        "flatrate": [{"provider_name": "Netflix"}],
                        "rent": [{"provider_name": "Amazon"}],
                        "buy": [{"provider_name": "Apple"}],
                    }
                }
            }
        }
        provs = _make_fetcher()._extract_providers(detail)
        assert {(p.name, p.provider_type.value) for p in provs} == {
            ("Netflix", "flatrate"),
            ("Amazon", "rent"),
            ("Apple", "buy"),
        }

    def test_extract_keywords(self):
        detail = {"keywords": {"keywords": [{"id": 1, "name": "alien"}]}}
        kws = _make_fetcher()._extract_keywords(detail)
        assert [k.tmdb_id for k in kws] == [1]
        assert kws[0].name == "alien"

    def test_extract_collection_present(self):
        detail = {"belongs_to_collection": {"id": 10, "name": "Arrival Trilogy"}}
        coll = _make_fetcher()._extract_collection(detail)
        assert coll.tmdb_id == 10
        assert coll.name == "Arrival Trilogy"

    def test_extract_collection_absent(self):
        assert _make_fetcher()._extract_collection({}) is None

    def test_extract_production_companies(self):
        detail = {
            "production_companies": [
                {"id": 1, "name": "Paramount", "origin_country": "US"},
                {"id": 2, "name": "Lionsgate"},
            ]
        }
        companies = _make_fetcher()._extract_production_companies(detail)
        assert [c.name for c in companies] == ["Paramount", "Lionsgate"]
        assert companies[0].origin_country == "US"
        assert companies[1].origin_country is None


class TestParseDetailResponse:
    def test_builds_details_with_defaults(self):
        fetcher = _make_fetcher()
        detail = {
            "id": 329865,
            "original_title": "Arrival",
            "poster_path": "/p.jpg",
            "release_date": "2016-11-11",
            "vote_average": 7.9,
            "genres": [{"id": 878, "name": "Sci-Fi"}],
            "adult": False,
            "overview": "Linguist meets aliens.",
            "runtime": 116,
            "videos": {"results": []},
            "credits": {"cast": [], "crew": []},
            "watch/providers": {"results": {}},
            "keywords": {"keywords": []},
            "production_companies": [],
        }

        details = fetcher.parse_detail_response(movie_db_id=1, detail_res=detail)

        assert details.movie_db_id == 1
        assert details.tmdb_id == 329865
        assert details.title == "Arrival"
        assert details.poster_url == "https://img//p.jpg"
        assert details.release_year == 2016
        assert details.rating == 7.9
        assert details.genres == ["Sci-Fi"]
        assert details.genre_tmdb_ids == [878]
        assert details.runtime == 116

    def test_handles_missing_poster_and_release(self):
        fetcher = _make_fetcher()
        detail = {
            "id": 1,
            "videos": {"results": []},
            "credits": {"cast": [], "crew": []},
            "watch/providers": {"results": {}},
            "keywords": {"keywords": []},
            "production_companies": [],
            "genres": [],
        }
        details = fetcher.parse_detail_response(movie_db_id=1, detail_res=detail)
        assert details.poster_url == ""
        assert details.release_year == 0


class TestFetchHttpMethods:
    async def test_search_movies_returns_top_10(self):
        fetcher = _make_fetcher()
        client = MagicMock()
        response = MagicMock()
        response.json.return_value = {"results": [{"id": i} for i in range(20)]}
        client.get = AsyncMock(return_value=response)
        fetcher._client = client

        results = await fetcher.search_movies("Arrival")

        assert len(results) == 10

    async def test_fetch_detail_by_id_returns_none_on_exception(self):
        fetcher = _make_fetcher()
        client = MagicMock()
        client.get = AsyncMock(side_effect=RuntimeError("boom"))
        fetcher._client = client

        assert await fetcher.fetch_detail_by_id(1) is None

    async def test_fetch_detail_by_id_returns_none_when_not_found(self):
        fetcher = _make_fetcher()
        client = MagicMock()
        response = MagicMock()
        response.json.return_value = {"status_code": 34}  # TMDB "not found" - no "id"
        client.get = AsyncMock(return_value=response)
        fetcher._client = client

        assert await fetcher.fetch_detail_by_id(1) is None

    async def test_fetch_detail_by_id_returns_payload(self):
        fetcher = _make_fetcher()
        client = MagicMock()
        response = MagicMock()
        response.json.return_value = {"id": 329865, "title": "Arrival"}
        client.get = AsyncMock(return_value=response)
        fetcher._client = client

        res = await fetcher.fetch_detail_by_id(329865)

        assert res["id"] == 329865

    async def test_fetch_detail_by_title_returns_none_when_search_empty(self):
        fetcher = _make_fetcher()
        client = MagicMock()
        response = MagicMock()
        response.json.return_value = {"results": []}
        client.get = AsyncMock(return_value=response)
        fetcher._client = client

        assert await fetcher.fetch_detail_by_title("Nonexistent (1999)") is None

    async def test_fetch_detail_by_title_fetches_top_result_by_id(self):
        fetcher = _make_fetcher()
        search_resp = MagicMock()
        search_resp.json.return_value = {"results": [{"id": 42}]}
        detail_resp = MagicMock()
        detail_resp.json.return_value = {"id": 42, "title": "Result"}
        client = MagicMock()
        client.get = AsyncMock(side_effect=[search_resp, detail_resp])
        fetcher._client = client

        res = await fetcher.fetch_detail_by_title("Matrix, The (1999)")

        assert res["id"] == 42
        # Two HTTP calls: search + detail
        assert client.get.await_count == 2


class TestMovieHydratorGetOrFetch:
    def _make_hydrator(self, db_session_factory, tmdb):
        h = MovieHydrator.__new__(MovieHydrator)
        h.db_session_factory = db_session_factory
        h.neo4j_driver = None
        h.tmdb = tmdb
        h._kg_semaphore = None
        return h

    @staticmethod
    def _session_factory(session):
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=None)
        return MagicMock(return_value=ctx)

    async def test_returns_existing_db_movie(self):
        session = MagicMock()
        factory = self._session_factory(session)

        existing_movie = MagicMock()
        existing_movie.tmdb_id = 123
        existing_movie.title = "Arrival"
        expected_details = MagicMock()

        tmdb = MagicMock()

        with (
            patch(
                "movie_recommender.services.recommender.pipeline.hydrator.main.get_movie_by_id",
                new_callable=AsyncMock,
                return_value=existing_movie,
            ),
            patch(
                "movie_recommender.services.recommender.pipeline.hydrator.main.movie_to_details",
                new_callable=AsyncMock,
                return_value=expected_details,
            ),
        ):
            h = self._make_hydrator(factory, tmdb)
            result = await h.get_or_fetch_movie(1, "Arrival")

        assert result is expected_details
        tmdb.fetch_detail_by_title = AsyncMock()
        tmdb.fetch_detail_by_title.assert_not_awaited()

    async def test_returns_none_when_tmdb_not_found(self):
        session = MagicMock()
        factory = self._session_factory(session)

        tmdb = MagicMock()
        tmdb.fetch_detail_by_title = AsyncMock(return_value=None)

        with patch(
            "movie_recommender.services.recommender.pipeline.hydrator.main.get_movie_by_id",
            new_callable=AsyncMock,
            return_value=None,
        ):
            h = self._make_hydrator(factory, tmdb)
            assert await h.get_or_fetch_movie(1, "Nope (1999)") is None

    async def test_fetches_from_tmdb_and_saves(self):
        session = MagicMock()
        factory = self._session_factory(session)

        tmdb = MagicMock()
        tmdb.fetch_detail_by_title = AsyncMock(return_value={"id": 42})
        parsed = MagicMock()
        parsed.title = "Arrival"
        tmdb.parse_detail_response = MagicMock(return_value=parsed)

        with (
            patch(
                "movie_recommender.services.recommender.pipeline.hydrator.main.get_movie_by_id",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "movie_recommender.services.recommender.pipeline.hydrator.main.save_hydrated_movie",
                new_callable=AsyncMock,
            ) as mock_save,
        ):
            h = self._make_hydrator(factory, tmdb)
            result = await h.get_or_fetch_movie(1, "Arrival (2016)")

        assert result is parsed
        mock_save.assert_awaited_once_with(session, 1, parsed)

    async def test_swallows_save_errors(self):
        session = MagicMock()
        factory = self._session_factory(session)

        tmdb = MagicMock()
        tmdb.fetch_detail_by_title = AsyncMock(return_value={"id": 42})
        parsed = MagicMock()
        parsed.title = "Arrival"
        tmdb.parse_detail_response = MagicMock(return_value=parsed)

        with (
            patch(
                "movie_recommender.services.recommender.pipeline.hydrator.main.get_movie_by_id",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "movie_recommender.services.recommender.pipeline.hydrator.main.save_hydrated_movie",
                new_callable=AsyncMock,
                side_effect=RuntimeError("unique constraint"),
            ),
        ):
            h = self._make_hydrator(factory, tmdb)
            # Must not raise despite save failure.
            result = await h.get_or_fetch_movie(1, "Arrival (2016)")
            assert result is parsed

    def test_kg_semaphore_lazy_init(self):
        h = MovieHydrator.__new__(MovieHydrator)
        h._kg_semaphore = None
        sem = h._get_kg_semaphore()
        assert sem is h._get_kg_semaphore()
