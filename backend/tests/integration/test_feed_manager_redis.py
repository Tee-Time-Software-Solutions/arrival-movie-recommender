"""
FeedManager integration tests against a real Redis.

Validates end-to-end behaviour of the FeedManager's Redis-backed queue:
- Refill populates the queue from recommender output.
- Draining pops entries in FIFO order.
- Cross-user isolation (two user ids never see each other's queue).
- Flush clears a user's queue.
- ``_pop_unseen`` correctly skips movies in the seen set.

Recommender and Hydrator collaborators are replaced with lightweight async
stubs because this test intentionally exercises the Redis plumbing in
isolation — the recommender ML pipeline is already covered by the existing
``test_online_recommender.py`` integration test.

Gated by the ``real_redis`` fixture in ``conftest_integration`` so runs
without a reachable Redis skip cleanly.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest_plugins = ["tests.integration.conftest_integration"]

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_feed_manager(real_redis, movie_ids, titles=None):
    """Construct a FeedManager wired to the real Redis client.

    Recommender / Hydrator are stubbed so the test focuses on queue behaviour
    rather than ML + TMDB calls.

    We bypass ``FeedManager.__init__`` because it instantiates
    ``AppSettings()``, which requires a fully populated env in the process.
    Tests of Redis queue semantics shouldn't need TMDB/Firebase env vars.
    """

    from movie_recommender.services.recommender.pipeline.feed_manager.main import (
        FeedManager,
    )

    titles = titles or {mid: f"Movie {mid}" for mid in movie_ids}

    # Recommender stub — only the attributes FeedManager actually reads.
    recommender = SimpleNamespace(
        user_seen_movie_ids={},
        model_artifacts=SimpleNamespace(
            movie_id_to_index={mid: idx for idx, mid in enumerate(movie_ids)},
            movie_id_to_title=titles,
            movie_id_to_genres={mid: [] for mid in movie_ids},
        ),
        get_top_n_recommendations=AsyncMock(return_value=list(movie_ids)),
    )

    # Hydrator stub — returns a truthy object so FeedManager pushes the entry.
    async def _fake_get_or_fetch(*args, **kwargs):
        mid = kwargs.get("movie_db_id", args[0] if args else None)
        title = kwargs.get("movie_title", args[1] if len(args) > 1 else "")
        return SimpleNamespace(id=mid, title=title, tmdb_id=None)

    hydrator = SimpleNamespace(
        get_or_fetch_movie=AsyncMock(side_effect=_fake_get_or_fetch)
    )

    # Minimal stand-in for AppSettings.app_logic used by refill_queue.
    fake_app_logic = SimpleNamespace(
        batch_size=15, queue_min_capacity=5, over_fetch_factor=2
    )

    fm = FeedManager.__new__(FeedManager)
    fm.recommender = recommender
    fm.hydrator = hydrator
    fm.redis_client = real_redis
    fm.neo4j_driver = None
    fm.db_session_factory = None
    fm.settings = SimpleNamespace(app_logic=fake_app_logic)
    return fm


async def _cleanup_user(real_redis, user_id):
    """Best-effort cleanup of all keys touched by a test user."""
    from movie_recommender.services.recommender.pipeline.feed_manager.main import (
        SEEN_KEY_PREFIX,
    )

    await real_redis.delete(f"feed:user:{user_id}")
    await real_redis.delete(f"{SEEN_KEY_PREFIX}{user_id}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_refill_queue_populates_from_recommender(real_redis):
    """FeedManager.refill_queue should push ranked movies into the user queue."""

    # Unique synthetic user id derived from the per-test prefix so parallel
    # runs don't collide on the "feed:user:{id}" key.
    user_id = abs(hash(real_redis.test_prefix)) % 10_000_000
    queue_key = f"feed:user:{user_id}"

    movie_ids = [101, 102, 103, 104]
    fm = _make_feed_manager(real_redis, movie_ids)

    try:
        await fm.refill_queue(user_id=user_id, queue_key=queue_key)

        length = await real_redis.llen(queue_key)
        # batch_size defaults to 15 from AppSettings; we only have 4 movies.
        assert length == len(movie_ids)

        # Verify ordering: first pushed = first popped (FIFO via lpop/rpush).
        import json

        popped = []
        while True:
            raw = await real_redis.lpop(queue_key)
            if raw is None:
                break
            mid, _title = json.loads(raw)
            popped.append(mid)

        assert popped == movie_ids
    finally:
        await _cleanup_user(real_redis, user_id)


async def test_get_next_movie_records_seen_set(real_redis):
    """get_next_movie should lpop from the queue and mark the movie as seen."""
    from movie_recommender.services.recommender.pipeline.feed_manager.main import (
        SEEN_KEY_PREFIX,
    )

    user_id = abs(hash(real_redis.test_prefix + "_unseen")) % 10_000_000
    queue_key = f"feed:user:{user_id}"

    movie_ids = [201, 202, 203]
    fm = _make_feed_manager(real_redis, movie_ids)

    try:
        await fm.refill_queue(user_id=user_id, queue_key=queue_key)
        assert await real_redis.llen(queue_key) == len(movie_ids)

        result = await fm.get_next_movie(user_id=user_id)

        assert result is not None
        # One item consumed from the head of the queue.
        assert await real_redis.llen(queue_key) == len(movie_ids) - 1
        # The consumed movie id is now in the seen set.
        seen_members = await real_redis.smembers(f"{SEEN_KEY_PREFIX}{user_id}")
        assert {int(m) for m in seen_members} == {201}
    finally:
        await _cleanup_user(real_redis, user_id)


async def test_flush_feed_clears_user_queue(real_redis):
    """flush_feed should delete the user's queue without touching others."""

    user_a = abs(hash(real_redis.test_prefix + "_a")) % 10_000_000
    user_b = user_a + 1
    queue_a = f"feed:user:{user_a}"
    queue_b = f"feed:user:{user_b}"

    fm_a = _make_feed_manager(real_redis, [301, 302])
    fm_b = _make_feed_manager(real_redis, [401, 402])

    try:
        await fm_a.refill_queue(user_id=user_a, queue_key=queue_a)
        await fm_b.refill_queue(user_id=user_b, queue_key=queue_b)

        assert await real_redis.llen(queue_a) == 2
        assert await real_redis.llen(queue_b) == 2

        await fm_a.flush_feed(user_a)

        assert await real_redis.llen(queue_a) == 0
        # User B's queue must be untouched.
        assert await real_redis.llen(queue_b) == 2
    finally:
        await _cleanup_user(real_redis, user_a)
        await _cleanup_user(real_redis, user_b)


async def test_cross_user_isolation(real_redis):
    """Two users simultaneously using the queue must not see each other's movies."""

    user_a = abs(hash(real_redis.test_prefix + "_iso_a")) % 10_000_000
    user_b = user_a + 7

    fm_a = _make_feed_manager(real_redis, [501, 502, 503])
    fm_b = _make_feed_manager(real_redis, [601, 602, 603])

    try:
        await fm_a.refill_queue(user_id=user_a, queue_key=f"feed:user:{user_a}")
        await fm_b.refill_queue(user_id=user_b, queue_key=f"feed:user:{user_b}")

        import json

        raw_a = await real_redis.lpop(f"feed:user:{user_a}")
        raw_b = await real_redis.lpop(f"feed:user:{user_b}")

        assert raw_a and raw_b
        mid_a, _ = json.loads(raw_a)
        mid_b, _ = json.loads(raw_b)

        assert mid_a in {501, 502, 503}
        assert mid_b in {601, 602, 603}
        assert mid_a != mid_b
    finally:
        await _cleanup_user(real_redis, user_a)
        await _cleanup_user(real_redis, user_b)


async def test_refill_is_idempotent_on_repeat(real_redis):
    """Calling refill twice deletes+repopulates (no duplicate accumulation)."""

    user_id = abs(hash(real_redis.test_prefix + "_idem")) % 10_000_000
    queue_key = f"feed:user:{user_id}"

    fm = _make_feed_manager(real_redis, [701, 702])

    try:
        await fm.refill_queue(user_id=user_id, queue_key=queue_key)
        first_len = await real_redis.llen(queue_key)

        await fm.refill_queue(user_id=user_id, queue_key=queue_key)
        second_len = await real_redis.llen(queue_key)

        assert first_len == second_len == 2
    finally:
        await _cleanup_user(real_redis, user_id)
