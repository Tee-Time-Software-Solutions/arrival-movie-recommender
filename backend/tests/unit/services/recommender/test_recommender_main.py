"""Unit tests for ``services.recommender.main.Recommender`` and user_state helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from movie_recommender.schemas.requests.interactions import SwipeAction
from movie_recommender.services.recommender.main import (
    SEEN_KEY_PREFIX,
    USER_VECTOR_KEY_PREFIX,
    Recommender,
)
from movie_recommender.services.recommender.pipeline.online.serving.user_state import (
    base_user_vector,
    cold_start_vector,
)


def _make_artifacts(user_id_to_index=None, user_embeddings=None) -> MagicMock:
    """Build a minimal RecommenderArtifacts-shaped stub."""
    artifacts = MagicMock()
    artifacts.user_id_to_index = user_id_to_index or {}
    if user_embeddings is None:
        user_embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    artifacts.user_embeddings = user_embeddings
    return artifacts


def _make_session_factory(session):
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return MagicMock(return_value=ctx)


def _make_recommender(
    *,
    artifacts=None,
    redis_client=None,
    db_session_factory=None,
    learning_rate: float = 0.05,
    norm_cap: float = 10.0,
    adaptive_learning_strength: float = 0.0,
    exploration_weight: float = 0.0,
    diversity_weight: float = 0.0,
) -> Recommender:
    rec = Recommender.__new__(Recommender)
    rec.learning_rate = learning_rate
    rec.norm_cap = norm_cap
    rec.adaptive_learning_strength = adaptive_learning_strength
    rec.exploration_weight = exploration_weight
    rec.diversity_weight = diversity_weight
    rec.model_artifacts = artifacts or _make_artifacts()
    rec._db_session_factory = db_session_factory or _make_session_factory(MagicMock())
    rec._redis = redis_client
    return rec


class TestUserState:
    def test_cold_start_is_mean_of_user_embeddings(self):
        artifacts = _make_artifacts(
            user_embeddings=np.array([[1.0, 1.0], [3.0, 5.0]], dtype=np.float32)
        )
        vec = cold_start_vector(artifacts)
        np.testing.assert_allclose(vec, [2.0, 3.0])

    def test_base_user_vector_known_user(self):
        artifacts = _make_artifacts(
            user_id_to_index={7: 1},
            user_embeddings=np.array([[1.0, 1.0], [3.0, 5.0]], dtype=np.float32),
        )
        vec = base_user_vector(artifacts, user_id=7)
        np.testing.assert_allclose(vec, [3.0, 5.0])

    def test_base_user_vector_unknown_user_falls_back_to_cold_start(self):
        artifacts = _make_artifacts(
            user_id_to_index={},
            user_embeddings=np.array([[1.0, 1.0], [3.0, 5.0]], dtype=np.float32),
        )
        vec = base_user_vector(artifacts, user_id=999)
        np.testing.assert_allclose(vec, [2.0, 3.0])


class TestSetRedis:
    def test_set_redis_stores_client(self):
        rec = _make_recommender()
        client = MagicMock()
        rec.set_redis(client)
        assert rec._redis is client


class TestGetUserVector:
    async def test_uses_redis_cache_when_present(self):
        redis_client = MagicMock()
        vec = np.array([1.5, 2.5], dtype=np.float32)
        redis_client.get = AsyncMock(return_value=vec.tobytes())

        rec = _make_recommender(redis_client=redis_client)

        result = await rec._get_user_vector(1)

        np.testing.assert_allclose(result, vec)
        redis_client.get.assert_awaited_once_with(f"{USER_VECTOR_KEY_PREFIX}1")

    async def test_falls_back_to_db_when_redis_miss(self):
        redis_client = MagicMock()
        redis_client.get = AsyncMock(return_value=None)
        redis_client.set = AsyncMock()

        stored_vec = np.array([7.0, 8.0], dtype=np.float32)

        with patch(
            "movie_recommender.services.recommender.main.get_user_vector",
            new_callable=AsyncMock,
            return_value=stored_vec,
        ):
            rec = _make_recommender(redis_client=redis_client)
            result = await rec._get_user_vector(1)

        np.testing.assert_allclose(result, stored_vec)
        redis_client.set.assert_awaited_once()

    async def test_falls_back_to_base_when_db_miss(self):
        artifacts = _make_artifacts(
            user_id_to_index={1: 0},
            user_embeddings=np.array([[4.0, 4.0]], dtype=np.float32),
        )

        with patch(
            "movie_recommender.services.recommender.main.get_user_vector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            rec = _make_recommender(artifacts=artifacts)
            result = await rec._get_user_vector(1)

        np.testing.assert_allclose(result, [4.0, 4.0])


class TestPersistVectorToDb:
    async def test_swallows_exceptions(self):
        rec = _make_recommender()
        vec = np.array([1.0], dtype=np.float32)

        with patch(
            "movie_recommender.services.recommender.main.save_user_vector",
            new_callable=AsyncMock,
            side_effect=RuntimeError("db down"),
        ) as mock_save:
            # Must not raise
            await rec._persist_vector_to_db(1, vec)

        mock_save.assert_awaited_once()


class TestGetTopNRecommendations:
    async def test_uses_seen_set_from_redis(self):
        redis_client = MagicMock()
        redis_client.get = AsyncMock(return_value=None)
        redis_client.set = AsyncMock()
        redis_client.smembers = AsyncMock(return_value={b"1", b"2"})
        redis_client.hgetall = AsyncMock(return_value={})

        rec = _make_recommender(
            artifacts=_make_artifacts(user_id_to_index={7: 0}),
            redis_client=redis_client,
        )

        with (
            patch(
                "movie_recommender.services.recommender.main.get_user_vector",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "movie_recommender.services.recommender.main.rank_movie_ids",
                return_value=[10, 20, 30],
            ) as mock_rank,
        ):
            result = await rec.get_top_n_recommendations(user_id=7, n=3)

        assert result == [10, 20, 30]
        call_kwargs = mock_rank.call_args.kwargs
        assert call_kwargs["seen_movie_ids"] == {1, 2}
        redis_client.smembers.assert_awaited_once_with(f"{SEEN_KEY_PREFIX}7")


class TestSetUserFeedback:
    async def test_updates_vector_writes_redis_and_schedules_db(self):
        redis_client = MagicMock()
        redis_client.get = AsyncMock(return_value=None)
        redis_client.set = AsyncMock()
        redis_client.incr = AsyncMock(return_value=1)
        redis_client.expire = AsyncMock()

        rec = _make_recommender(
            artifacts=_make_artifacts(user_id_to_index={7: 0}),
            redis_client=redis_client,
        )

        updated = np.array([9.0, 9.0], dtype=np.float32)

        with (
            patch(
                "movie_recommender.services.recommender.main.get_user_vector",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "movie_recommender.services.recommender.main.apply_feedback_update",
                return_value=updated,
            ),
            patch.object(
                Recommender, "_persist_vector_to_db", new_callable=AsyncMock
            ) as mock_persist,
        ):
            await rec.set_user_feedback(
                user_id=7,
                movie_id=1,
                interaction_type=SwipeAction.LIKE,
                is_supercharged=False,
            )

        redis_client.set.assert_any_await(
            f"{USER_VECTOR_KEY_PREFIX}7", updated.tobytes()
        )
        # Background task was scheduled
        import asyncio

        await asyncio.sleep(0)
        mock_persist.assert_awaited()

    async def test_no_op_when_feedback_returns_none(self):
        rec = _make_recommender()

        with (
            patch(
                "movie_recommender.services.recommender.main.get_user_vector",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "movie_recommender.services.recommender.main.apply_feedback_update",
                return_value=None,
            ),
            patch.object(
                Recommender, "_persist_vector_to_db", new_callable=AsyncMock
            ) as mock_persist,
        ):
            await rec.set_user_feedback(
                user_id=7,
                movie_id=1,
                interaction_type=SwipeAction.DISLIKE,
                is_supercharged=False,
            )

        mock_persist.assert_not_awaited()
