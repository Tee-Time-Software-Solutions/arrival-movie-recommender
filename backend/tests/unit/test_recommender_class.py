import numpy as np
import pytest

from movie_recommender.schemas.interactions import SwipeAction
from movie_recommender.services.recommender.main import Recommender
from movie_recommender.services.recommender.serving.artifact_loader import RecommenderArtifacts


class TestRequireArtifacts:
    def test_raises_when_no_artifacts(self):
        rec = Recommender.__new__(Recommender)
        rec.artifacts = None
        rec._artifact_load_error = "file missing"
        rec.online_user_vectors = {}
        rec.user_seen_movie_ids = {}
        rec.eta = 0.05
        rec.norm_cap = 10.0

        with pytest.raises(RuntimeError, match="not available"):
            rec._require_artifacts()

    def test_returns_artifacts_when_loaded(self, recommender):
        result = recommender._require_artifacts()
        assert result is recommender.artifacts

    def test_error_message_includes_load_details(self):
        rec = Recommender.__new__(Recommender)
        rec.artifacts = None
        rec._artifact_load_error = "specific_file_missing.npy"
        rec.online_user_vectors = {}
        rec.user_seen_movie_ids = {}
        rec.eta = 0.05
        rec.norm_cap = 10.0

        with pytest.raises(RuntimeError, match="specific_file_missing.npy"):
            rec._require_artifacts()


class TestColdStartVector:
    def test_cold_start_is_mean_of_all_users(self, recommender, synthetic_artifacts):
        expected = synthetic_artifacts.user_embeddings.mean(axis=0)
        result = recommender._cold_start_vector(synthetic_artifacts)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_cold_start_dtype_is_float32(self, recommender, synthetic_artifacts):
        result = recommender._cold_start_vector(synthetic_artifacts)
        assert result.dtype == np.float32

    def test_cold_start_shape_matches_embedding_dim(self, recommender, synthetic_artifacts):
        result = recommender._cold_start_vector(synthetic_artifacts)
        assert result.shape == (4,)


class TestBaseUserVector:
    def test_known_user_returns_their_embedding(self, recommender, synthetic_artifacts):
        """User 1 (action fan) should get embedding [1, 0, 0, 0]."""
        result = recommender._base_user_vector("1")
        np.testing.assert_array_equal(result, synthetic_artifacts.user_embeddings[0])

    def test_known_user_2_returns_comedy_embedding(self, recommender, synthetic_artifacts):
        """User 2 (comedy fan) should get embedding [0, 1, 0, 0]."""
        result = recommender._base_user_vector("2")
        np.testing.assert_array_equal(result, synthetic_artifacts.user_embeddings[1])

    def test_unknown_user_returns_cold_start(self, recommender, synthetic_artifacts):
        result = recommender._base_user_vector("99999")
        expected = synthetic_artifacts.user_embeddings.mean(axis=0)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_non_numeric_user_returns_cold_start(self, recommender, synthetic_artifacts):
        result = recommender._base_user_vector("abc")
        expected = synthetic_artifacts.user_embeddings.mean(axis=0)
        np.testing.assert_allclose(result, expected, atol=1e-6)


class TestCurrentUserVector:
    def test_returns_base_when_no_online_vector(self, recommender, synthetic_artifacts):
        result = recommender._current_user_vector("1")
        np.testing.assert_array_equal(result, synthetic_artifacts.user_embeddings[0])

    def test_returns_online_vector_when_present(self, recommender):
        custom = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        recommender.online_user_vectors["1"] = custom
        result = recommender._current_user_vector("1")
        np.testing.assert_array_equal(result, custom)

    def test_online_vector_takes_precedence_over_base(self, recommender, synthetic_artifacts):
        """Even for a known user, online vector should override base embedding."""
        custom = np.array([9.0, 9.0, 9.0, 9.0], dtype=np.float32)
        recommender.online_user_vectors["1"] = custom
        result = recommender._current_user_vector("1")
        np.testing.assert_array_equal(result, custom)
        # Verify it is NOT the base embedding
        assert not np.array_equal(result, synthetic_artifacts.user_embeddings[0])


class TestGetTopN:
    def test_returns_correct_count(self, recommender):
        recs = recommender.get_top_n("1", n=3, user_preferences=None)
        assert len(recs) == 3

    def test_returns_tuples_of_int_and_str(self, recommender):
        recs = recommender.get_top_n("1", n=2, user_preferences=None)
        for movie_id, title in recs:
            assert isinstance(movie_id, int)
            assert isinstance(title, str)

    def test_action_fan_gets_action_movies_first(self, recommender):
        """User 1 (action fan [1,0,0,0]) should get movies with highest action scores first.
        Movie 100 [1,0,0,0] -> score 1.0
        Movie 103 [1,1,0,0] -> score 1.0
        These two should appear in top 2."""
        recs = recommender.get_top_n("1", n=2, user_preferences=None)
        rec_ids = {mid for mid, _ in recs}
        assert rec_ids == {100, 103}

    def test_comedy_fan_gets_comedy_movies_first(self, recommender):
        """User 2 (comedy fan [0,1,0,0]) should rank comedy movies highest.
        Movie 101 [0,1,0,0] -> score 1.0
        Movie 103 [1,1,0,0] -> score 1.0"""
        recs = recommender.get_top_n("2", n=2, user_preferences=None)
        rec_ids = {mid for mid, _ in recs}
        assert rec_ids == {101, 103}

    def test_all_movies_seen_returns_empty(self, recommender):
        recommender.user_seen_movie_ids["1"] = {100, 101, 102, 103, 104}
        recs = recommender.get_top_n("1", n=5, user_preferences=None)
        assert recs == []

    def test_n_larger_than_available_returns_all_unseen(self, recommender):
        recommender.user_seen_movie_ids["1"] = {100, 101, 102}
        recs = recommender.get_top_n("1", n=10, user_preferences=None)
        assert len(recs) == 2  # only 103 and 104 remain

    def test_seen_movies_excluded_from_results(self, recommender):
        recommender.user_seen_movie_ids["1"] = {100}
        recs = recommender.get_top_n("1", n=5, user_preferences=None)
        rec_ids = [mid for mid, _ in recs]
        assert 100 not in rec_ids

    def test_n_zero_returns_empty(self, recommender):
        recs = recommender.get_top_n("1", n=0, user_preferences=None)
        assert recs == []


class TestUpdateUser:
    def test_like_creates_online_vector(self, recommender):
        recommender.update_user("1", movie_id=100, action_type=SwipeAction.LIKE)
        assert "1" in recommender.online_user_vectors

    def test_like_adds_to_seen_set(self, recommender):
        recommender.update_user("1", movie_id=100, action_type=SwipeAction.LIKE)
        assert 100 in recommender.user_seen_movie_ids["1"]

    def test_skip_marks_seen_but_vector_unchanged(self, recommender, synthetic_artifacts):
        vec_before = recommender._current_user_vector("1").copy()
        recommender.update_user("1", movie_id=100, action_type=SwipeAction.SKIP)

        assert 100 in recommender.user_seen_movie_ids["1"]
        # Skip preference is 0, so update_user_vector returns a copy unchanged
        # But the online vector IS stored (because the code always stores after update)
        if "1" in recommender.online_user_vectors:
            np.testing.assert_array_almost_equal(
                recommender.online_user_vectors["1"], vec_before
            )

    def test_unknown_movie_marks_seen_no_vector_update(self, recommender):
        recommender.update_user("1", movie_id=999, action_type=SwipeAction.LIKE)
        assert 999 in recommender.user_seen_movie_ids["1"]
        # Unknown movie_id returns early before updating vector
        assert "1" not in recommender.online_user_vectors

    def test_multiple_updates_accumulate_seen(self, recommender):
        recommender.update_user("1", movie_id=100, action_type=SwipeAction.LIKE)
        recommender.update_user("1", movie_id=101, action_type=SwipeAction.DISLIKE)
        assert recommender.user_seen_movie_ids["1"] == {100, 101}

    def test_dislike_moves_vector_away(self, recommender, synthetic_artifacts):
        """Disliking action movie should decrease action score for user 1."""
        vec_before = recommender._current_user_vector("1").copy()
        movie_idx = synthetic_artifacts.movie_id_to_index[100]
        score_before = float(synthetic_artifacts.movie_embeddings[movie_idx] @ vec_before)

        recommender.update_user("1", movie_id=100, action_type=SwipeAction.DISLIKE)

        vec_after = recommender._current_user_vector("1")
        score_after = float(synthetic_artifacts.movie_embeddings[movie_idx] @ vec_after)
        assert score_after < score_before
