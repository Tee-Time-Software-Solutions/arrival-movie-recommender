"""Unit tests for FeedManager service.

Covers queue pop/refill behavior, per-user Redis key generation, seen-set
deduplication, refill threshold triggering, and flush behavior. All external
dependencies (Redis, recommender, hydrator, DB) are mocked — no real I/O.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from movie_recommender.schemas.requests.movies import MovieDetails
from movie_recommender.services.feed_manager.main import FeedManager, SEEN_KEY_PREFIX


def _make_movie_details(movie_db_id: int = 1, tmdb_id: int = 100) -> MovieDetails:
    return MovieDetails(
        movie_db_id=movie_db_id,
        tmdb_id=tmdb_id,
        title="Arrival",
        poster_url="https://img/arrival.jpg",
        release_year=2016,
        rating=7.9,
        genres=["Sci-Fi"],
        is_adult=False,
        synopsis="Linguist meets aliens.",
        runtime=116,
    )


def _make_feed_manager(
    *,
    redis_client=None,
    recommender=None,
    hydrator=None,
    neo4j_driver=None,
    db_session_factory=None,
    batch_size: int = 15,
    queue_min_capacity: int = 5,
) -> FeedManager:
    """Construct a FeedManager bypassing __init__ (which would call AppSettings())."""
    fm = FeedManager.__new__(FeedManager)
    fm.redis_client = redis_client or AsyncMock()
    fm.recommender = recommender or MagicMock()
    fm.hydrator = hydrator or MagicMock()
    fm.neo4j_driver = neo4j_driver
    fm.db_session_factory = db_session_factory
    fm.settings = SimpleNamespace(
        app_logic=SimpleNamespace(
            batch_size=batch_size,
            queue_min_capacity=queue_min_capacity,
        )
    )
    return fm


class TestFlushFeed:
    @pytest.mark.asyncio
    async def test_flush_feed_deletes_user_queue_key(self):
        redis_client = AsyncMock()
        fm = _make_feed_manager(redis_client=redis_client)

        await fm.flush_feed(user_id=42)

        redis_client.delete.assert_awaited_once_with("feed:user:42")

    @pytest.mark.asyncio
    async def test_flush_feed_uses_distinct_keys_per_user(self):
        redis_client = AsyncMock()
        fm = _make_feed_manager(redis_client=redis_client)

        await fm.flush_feed(user_id=1)
        await fm.flush_feed(user_id=2)

        keys = [c.args[0] for c in redis_client.delete.await_args_list]
        assert keys == ["feed:user:1", "feed:user:2"]


class TestPopUnseen:
    @pytest.mark.asyncio
    async def test_returns_none_when_queue_empty(self):
        redis_client = AsyncMock()
        redis_client.lpop.return_value = None
        fm = _make_feed_manager(redis_client=redis_client)

        result = await fm._pop_unseen("feed:user:1", seen_ids=set())

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_first_unseen_entry(self):
        redis_client = AsyncMock()
        redis_client.lpop.return_value = json.dumps([11, "Movie 11"])
        fm = _make_feed_manager(redis_client=redis_client)

        raw = await fm._pop_unseen("feed:user:1", seen_ids=set())

        assert json.loads(raw) == [11, "Movie 11"]

    @pytest.mark.asyncio
    async def test_skips_seen_entries_and_returns_next_unseen(self):
        redis_client = AsyncMock()
        redis_client.lpop.side_effect = [
            json.dumps([1, "seen 1"]),
            json.dumps([2, "seen 2"]),
            json.dumps([3, "fresh 3"]),
        ]
        fm = _make_feed_manager(redis_client=redis_client)

        raw = await fm._pop_unseen("feed:user:1", seen_ids={1, 2})

        assert json.loads(raw) == [3, "fresh 3"]
        assert redis_client.lpop.await_count == 3

    @pytest.mark.asyncio
    async def test_returns_none_when_all_entries_seen(self):
        redis_client = AsyncMock()
        redis_client.lpop.side_effect = [
            json.dumps([1, "a"]),
            json.dumps([2, "b"]),
            None,
        ]
        fm = _make_feed_manager(redis_client=redis_client)

        result = await fm._pop_unseen("feed:user:1", seen_ids={1, 2})

        assert result is None


class TestGetNextMovie:
    @pytest.mark.asyncio
    async def test_happy_path_pops_and_hydrates(self):
        redis_client = AsyncMock()
        redis_client.smembers.return_value = set()
        redis_client.lpop.return_value = json.dumps([7, "Arrival"])
        # queue length is plenty → no async refill
        redis_client.llen.return_value = 100

        hydrator = MagicMock()
        hydrator.get_or_fetch_movie = AsyncMock(return_value=_make_movie_details())

        fm = _make_feed_manager(redis_client=redis_client, hydrator=hydrator)

        details = await fm.get_next_movie(user_id=42)

        assert details is not None
        assert details.title == "Arrival"
        hydrator.get_or_fetch_movie.assert_awaited_once_with(
            movie_db_id=7, movie_title="Arrival"
        )
        # User's seen set is updated synchronously
        redis_client.sadd.assert_awaited_once_with(f"{SEEN_KEY_PREFIX}42", 7)

    @pytest.mark.asyncio
    async def test_skips_seen_movie_from_redis_set(self):
        redis_client = AsyncMock()
        redis_client.smembers.return_value = {b"7"}
        # first pop returns a seen movie, second returns a fresh one
        redis_client.lpop.side_effect = [
            json.dumps([7, "Seen"]),
            json.dumps([8, "Fresh"]),
        ]
        redis_client.llen.return_value = 100

        hydrator = MagicMock()
        hydrator.get_or_fetch_movie = AsyncMock(
            return_value=_make_movie_details(movie_db_id=8)
        )

        fm = _make_feed_manager(redis_client=redis_client, hydrator=hydrator)

        details = await fm.get_next_movie(user_id=42)

        assert details is not None
        # The hydrator is called for the fresh (unseen) movie id
        hydrator.get_or_fetch_movie.assert_awaited_once_with(
            movie_db_id=8, movie_title="Fresh"
        )

    @pytest.mark.asyncio
    async def test_refills_when_queue_empty(self):
        redis_client = AsyncMock()
        redis_client.smembers.return_value = set()
        # first pop: empty → refill → pop again: success
        redis_client.lpop.side_effect = [None, json.dumps([5, "After refill"])]
        redis_client.llen.return_value = 100

        hydrator = MagicMock()
        hydrator.get_or_fetch_movie = AsyncMock(return_value=_make_movie_details())

        fm = _make_feed_manager(redis_client=redis_client, hydrator=hydrator)
        fm.refill_queue = AsyncMock()

        details = await fm.get_next_movie(user_id=42)

        assert details is not None
        fm.refill_queue.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_refill_yields_nothing(self):
        redis_client = AsyncMock()
        redis_client.smembers.return_value = set()
        redis_client.lpop.return_value = None  # always empty
        redis_client.llen.return_value = 0

        fm = _make_feed_manager(redis_client=redis_client)
        fm.refill_queue = AsyncMock()

        details = await fm.get_next_movie(user_id=42)

        assert details is None
        # sadd never called because no movie was served
        redis_client.sadd.assert_not_called()

    @pytest.mark.asyncio
    async def test_triggers_background_refill_below_threshold(self):
        redis_client = AsyncMock()
        redis_client.smembers.return_value = set()
        redis_client.lpop.return_value = json.dumps([9, "Low stock"])
        # below queue_min_capacity of 5
        redis_client.llen.return_value = 2

        hydrator = MagicMock()
        hydrator.get_or_fetch_movie = AsyncMock(return_value=_make_movie_details())

        fm = _make_feed_manager(
            redis_client=redis_client, hydrator=hydrator, queue_min_capacity=5
        )

        with patch(
            "movie_recommender.services.feed_manager.main.asyncio.create_task"
        ) as mock_create_task:
            details = await fm.get_next_movie(user_id=42)

        assert details is not None
        mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_background_refill_when_queue_healthy(self):
        redis_client = AsyncMock()
        redis_client.smembers.return_value = set()
        redis_client.lpop.return_value = json.dumps([9, "Healthy"])
        redis_client.llen.return_value = 50

        hydrator = MagicMock()
        hydrator.get_or_fetch_movie = AsyncMock(return_value=_make_movie_details())

        fm = _make_feed_manager(
            redis_client=redis_client, hydrator=hydrator, queue_min_capacity=5
        )

        with patch(
            "movie_recommender.services.feed_manager.main.asyncio.create_task"
        ) as mock_create_task:
            await fm.get_next_movie(user_id=42)

        mock_create_task.assert_not_called()


class TestRefillQueue:
    @pytest.mark.asyncio
    async def test_refill_populates_queue_with_ranked_movies(self):
        redis_client = AsyncMock()

        recommender = MagicMock()
        recommender.artifacts = SimpleNamespace(
            movie_id_to_index={100: 0, 101: 1, 102: 2},
            movie_id_to_title={100: "A", 101: "B", 102: "C"},
        )
        recommender.user_seen_movie_ids = {42: {999}}
        recommender.get_top_n_recommendations = AsyncMock(return_value=[101, 100, 102])

        hydrator = MagicMock()
        hydrator.get_or_fetch_movie = AsyncMock(
            side_effect=[
                _make_movie_details(movie_db_id=101),
                _make_movie_details(movie_db_id=100),
                _make_movie_details(movie_db_id=102),
            ]
        )

        fm = _make_feed_manager(
            redis_client=redis_client,
            recommender=recommender,
            hydrator=hydrator,
            batch_size=15,
        )

        await fm.refill_queue(user_id=42, queue_key="feed:user:42")

        # Stale queue is cleared before refill
        redis_client.delete.assert_awaited_once_with("feed:user:42")
        # Cached seen set for this user was invalidated
        assert 42 not in recommender.user_seen_movie_ids
        # 3 rpush calls, one per hydrated movie
        assert redis_client.rpush.await_count == 3
        pushed_payloads = [
            json.loads(c.args[1]) for c in redis_client.rpush.await_args_list
        ]
        assert [p[0] for p in pushed_payloads] == [101, 100, 102]

    @pytest.mark.asyncio
    async def test_refill_respects_batch_size(self):
        redis_client = AsyncMock()
        recommender = MagicMock()
        recommender.artifacts = SimpleNamespace(
            movie_id_to_index={i: i for i in range(20)},
            movie_id_to_title={i: f"M{i}" for i in range(20)},
        )
        recommender.user_seen_movie_ids = {}
        recommender.get_top_n_recommendations = AsyncMock(
            return_value=list(range(20))
        )

        hydrator = MagicMock()
        hydrator.get_or_fetch_movie = AsyncMock(
            side_effect=[_make_movie_details() for _ in range(3)]
        )

        fm = _make_feed_manager(
            redis_client=redis_client,
            recommender=recommender,
            hydrator=hydrator,
            batch_size=3,
        )

        await fm.refill_queue(user_id=1, queue_key="feed:user:1")

        # Only batch_size (3) movies get hydrated
        assert hydrator.get_or_fetch_movie.await_count == 3

    @pytest.mark.asyncio
    async def test_refill_skips_failed_hydrations(self):
        redis_client = AsyncMock()
        recommender = MagicMock()
        recommender.artifacts = SimpleNamespace(
            movie_id_to_index={1: 0, 2: 1},
            movie_id_to_title={1: "One", 2: "Two"},
        )
        recommender.user_seen_movie_ids = {}
        recommender.get_top_n_recommendations = AsyncMock(return_value=[1, 2])

        hydrator = MagicMock()
        # First hydration returns None (failure), second returns a movie
        hydrator.get_or_fetch_movie = AsyncMock(
            side_effect=[None, _make_movie_details(movie_db_id=2)]
        )

        fm = _make_feed_manager(
            redis_client=redis_client,
            recommender=recommender,
            hydrator=hydrator,
            batch_size=5,
        )

        await fm.refill_queue(user_id=1, queue_key="feed:user:1")

        # Only one movie pushed (the second one, since the first failed)
        assert redis_client.rpush.await_count == 1
        pushed_payload = json.loads(redis_client.rpush.await_args_list[0].args[1])
        assert pushed_payload[0] == 2
