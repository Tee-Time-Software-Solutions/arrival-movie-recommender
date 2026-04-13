import numpy as np
import pytest

from movie_recommender.schemas.requests.interactions import SwipeAction
from movie_recommender.services.recommender.main import Recommender
from movie_recommender.services.recommender.pipeline.online.artifacts import (
    RecommenderArtifacts,
    require_model_artifacts,
)
from movie_recommender.services.recommender.pipeline.online.serving.user_state import (
    cold_start_vector,
)


async def _top_n(recommender: Recommender, user_id: int, n: int) -> list[tuple[int, str]]:
    ranked_ids = await recommender.get_top_n_recommendations(user_id=user_id, n=n)
    return [
        (mid, recommender.model_artifacts.movie_id_to_title.get(mid, f"movie_{mid}"))
        for mid in ranked_ids
    ]


class TestRequireModelArtifacts:
    def test_raises_when_no_artifacts(self):
        with pytest.raises(RuntimeError, match="not available"):
            require_model_artifacts(None, "file missing")

    def test_returns_artifacts_when_loaded(self, synthetic_artifacts):
        result = require_model_artifacts(synthetic_artifacts, None)
        assert result is synthetic_artifacts

    def test_error_message_includes_load_details(self):
        with pytest.raises(RuntimeError, match="specific_file_missing.npy"):
            require_model_artifacts(None, "specific_file_missing.npy")


class TestColdStartVector:
    def test_cold_start_is_mean_of_all_users(self, synthetic_artifacts):
        expected = synthetic_artifacts.user_embeddings.mean(axis=0)
        result = cold_start_vector(synthetic_artifacts)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_cold_start_dtype_is_float32(self, synthetic_artifacts):
        result = cold_start_vector(synthetic_artifacts)
        assert result.dtype == np.float32

    def test_cold_start_shape_matches_embedding_dim(self, synthetic_artifacts):
        result = cold_start_vector(synthetic_artifacts)
        assert result.shape == (4,)


class TestGetTopN:
    @pytest.mark.asyncio
    async def test_returns_correct_count(self, recommender):
        recs = await _top_n(recommender, 1, n=3)
        assert len(recs) == 3

    @pytest.mark.asyncio
    async def test_returns_tuples_of_int_and_str(self, recommender):
        recs = await _top_n(recommender, 1, n=2)
        for movie_id, title in recs:
            assert isinstance(movie_id, int)
            assert isinstance(title, str)

    @pytest.mark.asyncio
    async def test_all_movies_seen_returns_empty(self, recommender):
        recommender._redis.smembers.return_value = {"100", "101", "102", "103", "104"}
        recs = await _top_n(recommender, 1, n=5)
        assert recs == []

    @pytest.mark.asyncio
    async def test_seen_movies_excluded_from_results(self, recommender):
        recommender._redis.smembers.return_value = {"100"}
        recs = await _top_n(recommender, 1, n=5)
        rec_ids = [mid for mid, _ in recs]
        assert 100 not in rec_ids

    @pytest.mark.asyncio
    async def test_n_larger_than_available_returns_all_unseen(self, recommender):
        recommender._redis.smembers.return_value = {"100", "101", "102"}
        recs = await _top_n(recommender, 1, n=10)
        assert len(recs) == 2  # only 103 and 104 remain

    @pytest.mark.asyncio
    async def test_n_zero_returns_empty(self, recommender):
        recs = await _top_n(recommender, 1, n=0)
        assert recs == []


class TestUpdateUser:
    @pytest.mark.asyncio
    async def test_like_writes_vector_to_redis(self, recommender):
        await recommender.set_user_feedback(
            user_id=1, movie_id=100, interaction_type=SwipeAction.LIKE, is_supercharged=False
        )
        recommender._redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_movie_does_not_write_to_redis(self, recommender):
        await recommender.set_user_feedback(
            user_id=1, movie_id=999, interaction_type=SwipeAction.LIKE, is_supercharged=False
        )
        recommender._redis.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_like_moves_vector_toward_movie(self, recommender, synthetic_artifacts):
        movie_idx = synthetic_artifacts.movie_id_to_index[100]
        movie_vec = synthetic_artifacts.movie_embeddings[movie_idx]
        cold = cold_start_vector(synthetic_artifacts)
        score_before = float(movie_vec @ cold)

        await recommender.set_user_feedback(
            user_id=1, movie_id=100, interaction_type=SwipeAction.LIKE, is_supercharged=False
        )

        written_bytes = recommender._redis.set.call_args[0][1]
        written_vec = np.frombuffer(written_bytes, dtype=np.float32)
        score_after = float(movie_vec @ written_vec)

        assert score_after > score_before

    @pytest.mark.asyncio
    async def test_dislike_moves_vector_away_from_movie(self, recommender, synthetic_artifacts):
        movie_idx = synthetic_artifacts.movie_id_to_index[100]
        movie_vec = synthetic_artifacts.movie_embeddings[movie_idx]
        cold = cold_start_vector(synthetic_artifacts)
        score_before = float(movie_vec @ cold)

        await recommender.set_user_feedback(
            user_id=1, movie_id=100, interaction_type=SwipeAction.DISLIKE, is_supercharged=False
        )

        written_bytes = recommender._redis.set.call_args[0][1]
        written_vec = np.frombuffer(written_bytes, dtype=np.float32)
        score_after = float(movie_vec @ written_vec)

        assert score_after < score_before

    @pytest.mark.asyncio
    async def test_like_triggers_db_persistence(self, recommender):
        await recommender.set_user_feedback(
            user_id=1, movie_id=100, interaction_type=SwipeAction.LIKE, is_supercharged=False
        )
        recommender._persist_vector_to_db.assert_called_once()

    @pytest.mark.asyncio
    async def test_skip_does_not_increment_feedback_count(self, recommender):
        await recommender.set_user_feedback(
            user_id=1,
            movie_id=100,
            interaction_type=SwipeAction.SKIP,
            is_supercharged=False,
        )
        recommender._redis.incr.assert_not_called()

    @pytest.mark.asyncio
    async def test_fresh_user_gets_larger_update_when_adaptive_learning_enabled(
        self, recommender, synthetic_artifacts
    ):
        recommender.adaptive_learning_strength = 0.2

        movie_idx = synthetic_artifacts.movie_id_to_index[100]
        movie_vec = synthetic_artifacts.movie_embeddings[movie_idx]
        cold = cold_start_vector(synthetic_artifacts)
        recommender._redis.get.side_effect = [None, None]

        await recommender.set_user_feedback(
            user_id=1,
            movie_id=100,
            interaction_type=SwipeAction.LIKE,
            is_supercharged=False,
        )
        fresh_bytes = recommender._redis.set.call_args[0][1]
        fresh_vec = np.frombuffer(fresh_bytes, dtype=np.float32)
        fresh_delta = float(movie_vec @ (fresh_vec - cold))

        recommender._redis.set.reset_mock()
        recommender._redis.incr.reset_mock()
        recommender._redis.expire.reset_mock()
        recommender._redis.get.side_effect = [None, b"100"]

        await recommender.set_user_feedback(
            user_id=1,
            movie_id=100,
            interaction_type=SwipeAction.LIKE,
            is_supercharged=False,
        )
        mature_bytes = recommender._redis.set.call_args[0][1]
        mature_vec = np.frombuffer(mature_bytes, dtype=np.float32)
        mature_delta = float(movie_vec @ (mature_vec - cold))

        assert fresh_delta > mature_delta
