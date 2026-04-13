"""Unit tests for swipe_worker service.

Covers enqueue_swipe payload shape and drain_swipe_queue batching/idempotency/
error handling. Redis and DB are mocked — no real I/O, no sleeping loops.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from movie_recommender.services.swipe_worker.main import (
    SWIPE_QUEUE_KEY,
    drain_swipe_queue,
    enqueue_swipe,
)


class _FakeDBSessionFactory:
    """Async-context-manager factory that yields a shared mock DB session."""

    def __init__(self):
        self.db = MagicMock()
        self.calls = 0

    def __call__(self):
        self.calls += 1

        @asynccontextmanager
        async def _cm():
            yield self.db

        return _cm()


class TestEnqueueSwipe:
    @pytest.mark.asyncio
    async def test_pushes_json_payload_to_swipe_queue(self):
        redis_client = AsyncMock()

        await enqueue_swipe(
            redis_client=redis_client,
            user_id=1,
            movie_id=42,
            action_type="like",
            is_supercharged=False,
        )

        redis_client.rpush.assert_awaited_once()
        key, payload = redis_client.rpush.await_args.args
        assert key == SWIPE_QUEUE_KEY
        assert json.loads(payload) == {
            "user_id": 1,
            "movie_id": 42,
            "action_type": "like",
            "is_supercharged": False,
        }

    @pytest.mark.asyncio
    async def test_supercharged_flag_persists_in_payload(self):
        redis_client = AsyncMock()

        await enqueue_swipe(
            redis_client=redis_client,
            user_id=7,
            movie_id=99,
            action_type="dislike",
            is_supercharged=True,
        )

        payload = json.loads(redis_client.rpush.await_args.args[1])
        assert payload["is_supercharged"] is True
        assert payload["action_type"] == "dislike"


class TestDrainSwipeQueue:
    @pytest.mark.asyncio
    async def test_persists_batch_then_exits_on_cancel(self):
        """Worker drains a batch, calls create_swipe for each event, then loops."""
        redis_client = AsyncMock()
        # One full batch of 3 events, then queue empty forever
        events = [
            json.dumps(
                {
                    "user_id": 1,
                    "movie_id": 10,
                    "action_type": "like",
                    "is_supercharged": False,
                }
            ),
            json.dumps(
                {
                    "user_id": 1,
                    "movie_id": 11,
                    "action_type": "dislike",
                    "is_supercharged": False,
                }
            ),
            json.dumps(
                {
                    "user_id": 2,
                    "movie_id": 12,
                    "action_type": "like",
                    "is_supercharged": True,
                }
            ),
        ]
        # After the batch is drained, every subsequent lpop returns None
        redis_client.lpop.side_effect = events + [None] * 100

        factory = _FakeDBSessionFactory()

        with patch(
            "movie_recommender.services.swipe_worker.main.create_swipe",
            new_callable=AsyncMock,
        ) as mock_create_swipe:
            # Cancel after a brief moment so the infinite loop exits
            task = asyncio.create_task(
                drain_swipe_queue(
                    redis_client=redis_client,
                    db_session_factory=factory,
                    batch_size=10,
                    poll_interval=0.01,
                )
            )
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert mock_create_swipe.await_count == 3
        # Check the first persisted event kwargs
        first_call = mock_create_swipe.await_args_list[0]
        assert first_call.kwargs["user_id"] == 1
        assert first_call.kwargs["movie_id"] == 10
        assert first_call.kwargs["action_type"] == "like"
        assert first_call.kwargs["is_supercharged"] is False
        # DB session factory was entered at least once for the batch
        assert factory.calls >= 1

    @pytest.mark.asyncio
    async def test_stops_reading_at_batch_size_boundary(self):
        """batch_size caps how many events are popped per tick."""
        redis_client = AsyncMock()
        redis_client.lpop.return_value = json.dumps(
            {
                "user_id": 1,
                "movie_id": 1,
                "action_type": "like",
                "is_supercharged": False,
            }
        )

        factory = _FakeDBSessionFactory()

        with patch(
            "movie_recommender.services.swipe_worker.main.create_swipe",
            new_callable=AsyncMock,
        ):
            task = asyncio.create_task(
                drain_swipe_queue(
                    redis_client=redis_client,
                    db_session_factory=factory,
                    batch_size=5,
                    poll_interval=10,  # large interval: at most 1 tick before cancel
                )
            )
            # Give the worker a moment to process exactly one batch,
            # then cancel during the long poll_interval sleep
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        # Exactly batch_size lpops in the single completed tick
        assert redis_client.lpop.await_count == 5

    @pytest.mark.asyncio
    async def test_malformed_event_does_not_crash_worker(self):
        """An exception on a single event is logged; worker continues."""
        redis_client = AsyncMock()
        good_event = json.dumps(
            {
                "user_id": 1,
                "movie_id": 1,
                "action_type": "like",
                "is_supercharged": False,
            }
        )
        bad_event = json.dumps(
            {
                "user_id": 2,
                "movie_id": 2,
                "action_type": "like",
                "is_supercharged": False,
            }
        )
        redis_client.lpop.side_effect = [good_event, bad_event] + [None] * 100

        factory = _FakeDBSessionFactory()

        with patch(
            "movie_recommender.services.swipe_worker.main.create_swipe",
            new_callable=AsyncMock,
        ) as mock_create_swipe:
            # First call succeeds, second raises — loop must survive it
            mock_create_swipe.side_effect = [None, RuntimeError("boom")]

            task = asyncio.create_task(
                drain_swipe_queue(
                    redis_client=redis_client,
                    db_session_factory=factory,
                    batch_size=10,
                    poll_interval=0.01,
                )
            )
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        # Worker tried to persist both events despite the second failing
        assert mock_create_swipe.await_count == 2
        # Worker is still alive enough to have polled the empty queue more times
        assert redis_client.lpop.await_count > 2

    @pytest.mark.asyncio
    async def test_no_db_session_when_queue_empty(self):
        """When no events are fetched, the DB session factory is never opened."""
        redis_client = AsyncMock()
        redis_client.lpop.return_value = None

        factory = _FakeDBSessionFactory()

        with patch(
            "movie_recommender.services.swipe_worker.main.create_swipe",
            new_callable=AsyncMock,
        ) as mock_create_swipe:
            task = asyncio.create_task(
                drain_swipe_queue(
                    redis_client=redis_client,
                    db_session_factory=factory,
                    batch_size=5,
                    poll_interval=0.01,
                )
            )
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert factory.calls == 0
        mock_create_swipe.assert_not_called()
