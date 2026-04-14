"""Unit tests for ``services.recommender.pipeline.feed_manager.main.FeedManager``."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from movie_recommender.schemas.requests.movies import MovieDetails, MovieProvider
from movie_recommender.schemas.requests.users import UserPreferences
from movie_recommender.services.recommender.pipeline.feed_manager.main import (
    SEEN_KEY_PREFIX,
    FeedManager,
)


def _make_movie(
    movie_db_id: int = 1,
    title: str = "Arrival",
    genres=None,
    release_year: int = 2016,
    rating: float = 7.9,
    is_adult: bool = False,
    providers=None,
) -> MovieDetails:
    return MovieDetails(
        movie_db_id=movie_db_id,
        tmdb_id=329865,
        title=title,
        poster_url="https://img/p.jpg",
        release_year=release_year,
        rating=rating,
        genres=genres if genres is not None else ["Sci-Fi"],
        is_adult=is_adult,
        synopsis="Linguist meets aliens.",
        runtime=116,
        movie_providers=providers or [],
    )


def _make_fm(
    *,
    recommender=None,
    hydrator=None,
    redis_client=None,
    neo4j_driver=None,
    db_session_factory=None,
    batch_size: int = 2,
    queue_min_capacity: int = 5,
    over_fetch_factor: int = 2,
) -> FeedManager:
    fm = FeedManager.__new__(FeedManager)
    fm.recommender = recommender or MagicMock()
    fm.hydrator = hydrator or MagicMock()
    fm.redis_client = redis_client or MagicMock()
    fm.neo4j_driver = neo4j_driver
    fm.db_session_factory = db_session_factory
    fm.settings = SimpleNamespace(
        app_logic=SimpleNamespace(
            batch_size=batch_size,
            queue_min_capacity=queue_min_capacity,
            over_fetch_factor=over_fetch_factor,
        )
    )
    return fm


class TestFlushFeed:
    async def test_deletes_queue_key(self):
        redis_client = MagicMock()
        redis_client.delete = AsyncMock()
        fm = _make_fm(redis_client=redis_client)

        await fm.flush_feed(user_id=7)

        redis_client.delete.assert_awaited_once_with("feed:user:7")


class TestMatchesPreferences:
    def test_passes_with_no_filters(self):
        fm = _make_fm()
        assert fm._matches_preferences(_make_movie(), UserPreferences()) is True

    def test_included_genres_require_overlap(self):
        fm = _make_fm()
        movie = _make_movie(genres=["Drama"])
        prefs = UserPreferences(included_genres=["Sci-Fi"])
        assert fm._matches_preferences(movie, prefs) is False

    def test_excluded_genres_reject(self):
        fm = _make_fm()
        movie = _make_movie(genres=["Horror"])
        prefs = UserPreferences(excluded_genres=["Horror"])
        assert fm._matches_preferences(movie, prefs) is False

    def test_year_bounds(self):
        fm = _make_fm()
        movie = _make_movie(release_year=1995)
        assert (
            fm._matches_preferences(movie, UserPreferences(min_release_year=2000))
            is False
        )
        assert (
            fm._matches_preferences(movie, UserPreferences(max_release_year=1990))
            is False
        )

    def test_min_rating(self):
        fm = _make_fm()
        movie = _make_movie(rating=6.0)
        assert fm._matches_preferences(movie, UserPreferences(min_rating=7.0)) is False

    def test_adult_filter(self):
        fm = _make_fm()
        movie = _make_movie(is_adult=True)
        assert (
            fm._matches_preferences(movie, UserPreferences(include_adult=False))
            is False
        )

    def test_providers_require_overlap(self):
        fm = _make_fm()
        movie = _make_movie(
            providers=[MovieProvider(name="Disney+", provider_type="flatrate")]
        )
        prefs = UserPreferences(
            movie_providers=[MovieProvider(name="Netflix", provider_type="flatrate")]
        )
        assert fm._matches_preferences(movie, prefs) is False

    def test_all_filters_pass(self):
        fm = _make_fm()
        movie = _make_movie(
            genres=["Sci-Fi", "Drama"],
            release_year=2016,
            rating=8.0,
            providers=[MovieProvider(name="Netflix", provider_type="flatrate")],
        )
        prefs = UserPreferences(
            included_genres=["Sci-Fi"],
            excluded_genres=["Horror"],
            min_release_year=2000,
            max_release_year=2020,
            min_rating=7.0,
            include_adult=False,
            movie_providers=[MovieProvider(name="Netflix", provider_type="flatrate")],
        )
        assert fm._matches_preferences(movie, prefs) is True


class TestGetNextMovie:
    async def test_pops_and_hydrates_without_refill(self):
        redis_client = MagicMock()
        redis_client.lpop = AsyncMock(return_value=json.dumps([42, "Arrival"]))
        redis_client.llen = AsyncMock(return_value=10)
        redis_client.sadd = AsyncMock()

        hydrator = MagicMock()
        expected = _make_movie(movie_db_id=42)
        hydrator.get_or_fetch_movie = AsyncMock(return_value=expected)

        fm = _make_fm(redis_client=redis_client, hydrator=hydrator)

        result = await fm.get_next_movie(user_id=1)

        assert result is expected
        redis_client.lpop.assert_awaited_once_with("feed:user:1")
        redis_client.sadd.assert_awaited_once_with(f"{SEEN_KEY_PREFIX}1", 42)
        hydrator.get_or_fetch_movie.assert_awaited_once_with(
            movie_db_id=42, movie_title="Arrival"
        )

    async def test_refills_when_queue_empty(self):
        redis_client = MagicMock()
        redis_client.lpop = AsyncMock(side_effect=[None, json.dumps([5, "M"])])
        redis_client.sadd = AsyncMock()

        fm = _make_fm(redis_client=redis_client)
        fm.refill_queue = AsyncMock()

        hydrator = MagicMock()
        hydrator.get_or_fetch_movie = AsyncMock(return_value=_make_movie(movie_db_id=5))
        fm.hydrator = hydrator

        result = await fm.get_next_movie(user_id=1)

        fm.refill_queue.assert_awaited_once()
        assert result.movie_db_id == 5

    async def test_returns_none_when_refill_empty(self):
        redis_client = MagicMock()
        redis_client.lpop = AsyncMock(return_value=None)
        fm = _make_fm(redis_client=redis_client)
        fm.refill_queue = AsyncMock()

        result = await fm.get_next_movie(user_id=1)

        assert result is None

    async def test_background_refill_when_below_threshold(self):
        redis_client = MagicMock()
        redis_client.lpop = AsyncMock(return_value=json.dumps([9, "Nine"]))
        redis_client.llen = AsyncMock(return_value=1)  # below threshold (5)
        redis_client.sadd = AsyncMock()

        hydrator = MagicMock()
        hydrator.get_or_fetch_movie = AsyncMock(return_value=_make_movie(movie_db_id=9))

        fm = _make_fm(redis_client=redis_client, hydrator=hydrator)
        fm.refill_queue = AsyncMock()

        result = await fm.get_next_movie(user_id=1)

        assert result.movie_db_id == 9
        # Background task scheduled — allow it to run so the mock is awaited.
        await asyncio.sleep(0)
        fm.refill_queue.assert_awaited()


class TestRefillQueue:
    async def test_no_candidates_leaves_queue_empty(self):
        redis_client = MagicMock()
        redis_client.delete = AsyncMock()
        redis_client.rpush = AsyncMock()

        recommender = MagicMock()
        recommender.get_top_n_recommendations = AsyncMock(return_value=[])
        recommender.model_artifacts = MagicMock()
        recommender.model_artifacts.movie_id_to_title = {}

        fm = _make_fm(redis_client=redis_client, recommender=recommender)

        await fm.refill_queue(user_id=1, queue_key="feed:user:1")

        redis_client.delete.assert_awaited_once_with("feed:user:1")
        redis_client.rpush.assert_not_awaited()

    async def test_pushes_hydrated_movies(self):
        redis_client = MagicMock()
        redis_client.delete = AsyncMock()
        redis_client.rpush = AsyncMock()

        recommender = MagicMock()
        recommender.get_top_n_recommendations = AsyncMock(return_value=[1, 2])
        recommender.model_artifacts = MagicMock()
        recommender.model_artifacts.movie_id_to_title = {1: "A", 2: "B"}

        hydrator = MagicMock()
        hydrator.get_or_fetch_movie = AsyncMock(
            side_effect=[_make_movie(movie_db_id=1), _make_movie(movie_db_id=2)]
        )

        fm = _make_fm(
            redis_client=redis_client, recommender=recommender, hydrator=hydrator
        )

        await fm.refill_queue(user_id=1, queue_key="feed:user:1")

        assert redis_client.rpush.await_count == 1
        pushed_args = redis_client.rpush.await_args.args
        assert pushed_args[0] == "feed:user:1"
        assert json.loads(pushed_args[1]) == [1, "A"]
        assert json.loads(pushed_args[2]) == [2, "B"]

    async def test_skips_none_hydration_results(self):
        redis_client = MagicMock()
        redis_client.delete = AsyncMock()
        redis_client.rpush = AsyncMock()

        recommender = MagicMock()
        recommender.get_top_n_recommendations = AsyncMock(return_value=[1, 2])
        recommender.model_artifacts = MagicMock()
        recommender.model_artifacts.movie_id_to_title = {1: "A", 2: "B"}

        hydrator = MagicMock()
        hydrator.get_or_fetch_movie = AsyncMock(
            side_effect=[None, _make_movie(movie_db_id=2)]
        )

        fm = _make_fm(
            redis_client=redis_client, recommender=recommender, hydrator=hydrator
        )

        await fm.refill_queue(user_id=1, queue_key="feed:user:1")

        pushed_args = redis_client.rpush.await_args.args
        assert len(pushed_args) == 2  # key + 1 movie
        assert json.loads(pushed_args[1]) == [2, "B"]

    async def test_filters_movies_by_preferences(self):
        redis_client = MagicMock()
        redis_client.delete = AsyncMock()
        redis_client.rpush = AsyncMock()

        recommender = MagicMock()
        # Two candidates: one matches genre filter, one doesn't.
        recommender.get_top_n_recommendations = AsyncMock(return_value=[1, 2])
        recommender.model_artifacts = MagicMock()
        recommender.model_artifacts.movie_id_to_title = {1: "SF", 2: "Horror"}

        hydrator = MagicMock()
        hydrator.get_or_fetch_movie = AsyncMock(
            side_effect=[
                _make_movie(movie_db_id=1, genres=["Sci-Fi"]),
                _make_movie(movie_db_id=2, genres=["Horror"]),
            ]
        )

        fm = _make_fm(
            redis_client=redis_client, recommender=recommender, hydrator=hydrator
        )
        prefs = UserPreferences(included_genres=["Sci-Fi"])

        await fm.refill_queue(
            user_id=1, queue_key="feed:user:1", user_preferences=prefs
        )

        pushed_args = redis_client.rpush.await_args.args
        assert len(pushed_args) == 2  # only the Sci-Fi movie
        assert json.loads(pushed_args[1]) == [1, "SF"]

    async def test_stops_when_no_new_candidates(self):
        """Second recommender call returns the same ids — loop must break."""
        redis_client = MagicMock()
        redis_client.delete = AsyncMock()
        redis_client.rpush = AsyncMock()

        recommender = MagicMock()
        recommender.get_top_n_recommendations = AsyncMock(return_value=[1])
        recommender.model_artifacts = MagicMock()
        recommender.model_artifacts.movie_id_to_title = {1: "Horror"}

        hydrator = MagicMock()
        hydrator.get_or_fetch_movie = AsyncMock(
            return_value=_make_movie(movie_db_id=1, genres=["Horror"])
        )

        fm = _make_fm(
            redis_client=redis_client, recommender=recommender, hydrator=hydrator
        )
        prefs = UserPreferences(included_genres=["Sci-Fi"])  # will reject

        await fm.refill_queue(
            user_id=1, queue_key="feed:user:1", user_preferences=prefs
        )

        # Only one recommender round because new_ids is empty on the 2nd iter.
        assert recommender.get_top_n_recommendations.await_count == 2
        redis_client.rpush.assert_not_awaited()
