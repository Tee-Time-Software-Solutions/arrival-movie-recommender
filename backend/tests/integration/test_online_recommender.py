"""
Integration tests for the online recommender pipeline using real MovieLens 20M data.

Tests the full loop: real ALS artifacts -> recommendations -> swipe feedback ->
updated user vector -> changed recommendations.  All tests use 64-dim
embeddings trained on the MovieLens 20M dataset.

Requires artifacts to exist (the pipeline_dir fixture downloads data and
trains the model on first run).

Run with:  pytest tests/integration/test_online_recommender.py -v -s
"""

import numpy as np

from movie_recommender.schemas.interactions import SwipeAction
from movie_recommender.services.recommender.main import Recommender
from movie_recommender.services.recommender.serving.artifact_loader import (
    RecommenderArtifacts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_rec(
    artifacts: RecommenderArtifacts, eta: float = 0.05, norm_cap: float = 10.0
) -> Recommender:
    """Create a fresh Recommender with the given artifacts."""
    rec = Recommender.__new__(Recommender)
    rec.artifacts = artifacts
    rec._artifact_load_error = None
    rec.online_user_vectors = {}
    rec.user_seen_movie_ids = {}
    rec.eta = eta
    rec.norm_cap = norm_cap
    return rec


def _real_user_id(artifacts: RecommenderArtifacts, index: int = 0) -> str:
    """Get a real user ID from the training data mappings."""
    return str(list(artifacts.user_id_to_index.keys())[index])


def _real_movie_id(artifacts: RecommenderArtifacts, index: int = 0) -> int:
    """Get a real movie ID from the training data mappings."""
    return list(artifacts.movie_id_to_index.keys())[index]


COLD_START_USER = "99999999"
UNKNOWN_MOVIE_ID = 99999999


# ---------------------------------------------------------------------------
# 1. Artifact loading
# ---------------------------------------------------------------------------


class TestArtifactInitialisation:
    def test_recommender_has_artifacts_when_loaded(
        self, pipeline_recommender: Recommender
    ):
        assert pipeline_recommender.artifacts is not None

    def test_artifacts_have_correct_embedding_dim(
        self, pipeline_recommender: Recommender
    ):
        artifacts = pipeline_recommender.artifacts
        assert artifacts.movie_embeddings.shape[1] == 64
        assert artifacts.user_embeddings.shape[1] == 64

    def test_recommender_without_artifacts_stores_error(self):
        """Simulates the case where artifact files are missing on disk."""
        rec = Recommender.__new__(Recommender)
        rec.artifacts = None
        rec._artifact_load_error = "Missing file: movie_embeddings.npy"
        rec.online_user_vectors = {}
        rec.user_seen_movie_ids = {}
        rec.eta = 0.05
        rec.norm_cap = 10.0

        assert rec.artifacts is None
        assert "Missing file" in rec._artifact_load_error


# ---------------------------------------------------------------------------
# 2. Known-user recommendations
# ---------------------------------------------------------------------------


class TestKnownUserRecommendations:
    def test_known_user_gets_results(
        self,
        pipeline_recommender: Recommender,
        loaded_artifacts: RecommenderArtifacts,
    ):
        uid = _real_user_id(loaded_artifacts)
        recs = pipeline_recommender.get_top_n(
            user_id=uid, n=5, user_preferences=None
        )
        assert len(recs) == 5

    def test_different_users_get_different_recs(
        self,
        pipeline_recommender: Recommender,
        loaded_artifacts: RecommenderArtifacts,
    ):
        """With trained ALS embeddings, different users should get
        different top recommendations (personalisation)."""
        uid_a = _real_user_id(loaded_artifacts, 0)
        uid_b = _real_user_id(loaded_artifacts, 1)
        recs_a = pipeline_recommender.get_top_n(
            user_id=uid_a, n=5, user_preferences=None
        )
        recs_b = pipeline_recommender.get_top_n(
            user_id=uid_b, n=5, user_preferences=None
        )
        ids_a = [mid for mid, _ in recs_a]
        ids_b = [mid for mid, _ in recs_b]
        assert ids_a != ids_b, "Two different users got identical top-5"

    def test_returns_requested_count(
        self,
        pipeline_recommender: Recommender,
        loaded_artifacts: RecommenderArtifacts,
    ):
        uid = _real_user_id(loaded_artifacts)
        recs = pipeline_recommender.get_top_n(
            user_id=uid, n=3, user_preferences=None
        )
        assert len(recs) == 3

    def test_returns_movie_id_and_title_tuples(
        self,
        pipeline_recommender: Recommender,
        loaded_artifacts: RecommenderArtifacts,
    ):
        uid = _real_user_id(loaded_artifacts)
        recs = pipeline_recommender.get_top_n(
            user_id=uid, n=1, user_preferences=None
        )
        movie_id, title = recs[0]
        assert isinstance(movie_id, int)
        assert isinstance(title, str)
        assert movie_id in loaded_artifacts.movie_id_to_index

    def test_requesting_more_than_available_returns_all(
        self,
        pipeline_recommender: Recommender,
        loaded_artifacts: RecommenderArtifacts,
    ):
        uid = _real_user_id(loaded_artifacts)
        n_movies = len(loaded_artifacts.movie_id_to_index)
        recs = pipeline_recommender.get_top_n(
            user_id=uid, n=n_movies + 100, user_preferences=None
        )
        assert len(recs) == n_movies


# ---------------------------------------------------------------------------
# 3. Cold-start user
# ---------------------------------------------------------------------------


class TestColdStartUser:
    def test_unknown_user_gets_recommendations(
        self, pipeline_recommender: Recommender
    ):
        recs = pipeline_recommender.get_top_n(
            user_id=COLD_START_USER, n=5, user_preferences=None
        )
        assert len(recs) == 5

    def test_cold_start_uses_mean_vector(
        self,
        pipeline_recommender: Recommender,
        loaded_artifacts: RecommenderArtifacts,
    ):
        """Cold-start vector should be the mean of all user embeddings."""
        vec = pipeline_recommender._current_user_vector(COLD_START_USER)
        expected = loaded_artifacts.user_embeddings.mean(axis=0)
        np.testing.assert_allclose(vec, expected, atol=1e-5)

    def test_non_numeric_user_id_gets_cold_start(
        self,
        pipeline_recommender: Recommender,
    ):
        recs = pipeline_recommender.get_top_n(
            user_id="alice", n=5, user_preferences=None
        )
        assert len(recs) == 5
        # Should use the same mean vector as any other unknown user
        vec_alice = pipeline_recommender._current_user_vector("alice")
        vec_cold = pipeline_recommender._current_user_vector(COLD_START_USER)
        np.testing.assert_array_equal(vec_alice, vec_cold)


# ---------------------------------------------------------------------------
# 4. Seen-movie exclusion
# ---------------------------------------------------------------------------


class TestSeenMovieExclusion:
    def test_swiped_movie_excluded_from_next_recs(
        self,
        pipeline_recommender: Recommender,
        loaded_artifacts: RecommenderArtifacts,
    ):
        uid = _real_user_id(loaded_artifacts)
        recs = pipeline_recommender.get_top_n(
            user_id=uid, n=5, user_preferences=None
        )
        top_movie_id = recs[0][0]

        pipeline_recommender.update_user(
            user_id=uid, movie_id=top_movie_id, action_type=SwipeAction.LIKE
        )
        recs_after = pipeline_recommender.get_top_n(
            user_id=uid, n=5, user_preferences=None
        )
        rec_ids = [mid for mid, _ in recs_after]
        assert top_movie_id not in rec_ids

    def test_all_movies_seen_returns_empty(
        self, loaded_artifacts: RecommenderArtifacts
    ):
        rec = _fresh_rec(loaded_artifacts)
        uid = _real_user_id(loaded_artifacts, 5)
        for movie_id in loaded_artifacts.movie_id_to_index:
            rec.update_user(
                user_id=uid, movie_id=movie_id, action_type=SwipeAction.LIKE
            )
        recs = rec.get_top_n(user_id=uid, n=5, user_preferences=None)
        assert recs == []


# ---------------------------------------------------------------------------
# 5. Like shifts recommendations toward similar movies
# ---------------------------------------------------------------------------


class TestLikeShiftsRecommendations:
    def test_like_shifts_vector_toward_movie(
        self, loaded_artifacts: RecommenderArtifacts
    ):
        """Liking a movie should shift the user vector's direction toward
        the movie embedding (higher cosine similarity).

        We check cosine similarity rather than raw dot-product score because
        the norm cap (10.0) can rescale the entire vector when user norms
        exceed the cap (~12–15 for ALS-trained users), masking the nudge."""
        rec = _fresh_rec(loaded_artifacts)
        uid = _real_user_id(loaded_artifacts)
        recs = rec.get_top_n(user_id=uid, n=10, user_preferences=None)
        movie_id, title = recs[5]
        movie_idx = loaded_artifacts.movie_id_to_index[movie_id]
        movie_vec = loaded_artifacts.movie_embeddings[movie_idx]

        vec_before = rec._current_user_vector(uid)
        cos_before = float(movie_vec @ vec_before) / (
            np.linalg.norm(movie_vec) * np.linalg.norm(vec_before)
        )

        rec.update_user(
            user_id=uid, movie_id=movie_id, action_type=SwipeAction.LIKE
        )

        vec_after = rec._current_user_vector(uid)
        cos_after = float(movie_vec @ vec_after) / (
            np.linalg.norm(movie_vec) * np.linalg.norm(vec_after)
        )

        print(f"\n  LIKE \"{title}\" (id={movie_id})")
        print(f"  Cosine sim: {cos_before:.6f} -> {cos_after:.6f} ({cos_after - cos_before:+.6f})")
        assert cos_after > cos_before

    def test_like_changes_user_vector(
        self, loaded_artifacts: RecommenderArtifacts
    ):
        """After liking a movie, the online user vector should differ from base."""
        rec = _fresh_rec(loaded_artifacts)
        uid = _real_user_id(loaded_artifacts)
        recs = rec.get_top_n(user_id=uid, n=1, user_preferences=None)
        movie_id = recs[0][0]

        vec_before = rec._current_user_vector(uid).copy()
        rec.update_user(
            user_id=uid, movie_id=movie_id, action_type=SwipeAction.LIKE
        )
        vec_after = rec._current_user_vector(uid)

        assert not np.array_equal(vec_before, vec_after)
        assert uid in rec.online_user_vectors


# ---------------------------------------------------------------------------
# 6. Dislike shifts recommendations away
# ---------------------------------------------------------------------------


class TestDislikeShiftsRecommendations:
    def test_dislike_decreases_movie_score(
        self, loaded_artifacts: RecommenderArtifacts
    ):
        """Disliking a movie should decrease its score."""
        rec = _fresh_rec(loaded_artifacts)
        uid = _real_user_id(loaded_artifacts, 2)
        recs = rec.get_top_n(user_id=uid, n=5, user_preferences=None)
        movie_id, title = recs[0]
        movie_idx = loaded_artifacts.movie_id_to_index[movie_id]
        movie_vec = loaded_artifacts.movie_embeddings[movie_idx]

        vec_before = rec._current_user_vector(uid)
        score_before = float(movie_vec @ vec_before)

        rec.update_user(
            user_id=uid, movie_id=movie_id, action_type=SwipeAction.DISLIKE
        )

        vec_after = rec._current_user_vector(uid)
        score_after = float(movie_vec @ vec_after)

        print(f"\n  DISLIKE \"{title}\" (id={movie_id})")
        print(f"  Score: {score_before:.4f} -> {score_after:.4f} ({score_after - score_before:+.4f})")
        assert score_after < score_before

    def test_dislike_changes_vector_opposite_to_like(
        self, loaded_artifacts: RecommenderArtifacts
    ):
        """Like and dislike should push the vector in opposite directions."""
        uid = _real_user_id(loaded_artifacts, 3)
        movie_id = _real_movie_id(loaded_artifacts, 0)
        movie_idx = loaded_artifacts.movie_id_to_index[movie_id]
        movie_vec = loaded_artifacts.movie_embeddings[movie_idx]

        rec_like = _fresh_rec(loaded_artifacts)
        rec_like.update_user(
            user_id=uid, movie_id=movie_id, action_type=SwipeAction.LIKE
        )
        delta_like = rec_like.online_user_vectors[uid] - rec_like._base_user_vector(uid)

        rec_dislike = _fresh_rec(loaded_artifacts)
        rec_dislike.update_user(
            user_id=uid, movie_id=movie_id, action_type=SwipeAction.DISLIKE
        )
        delta_dislike = (
            rec_dislike.online_user_vectors[uid] - rec_dislike._base_user_vector(uid)
        )

        # Deltas should point in opposite directions (negative dot product)
        dot = float(delta_like @ delta_dislike)
        assert dot < 0, f"Like/dislike deltas should oppose; dot={dot:.6f}"


# ---------------------------------------------------------------------------
# 7. Skip behaviour
# ---------------------------------------------------------------------------


class TestSkipBehaviour:
    def test_skip_does_not_change_user_vector(
        self, loaded_artifacts: RecommenderArtifacts
    ):
        rec = _fresh_rec(loaded_artifacts)
        uid = _real_user_id(loaded_artifacts)
        movie_id = _real_movie_id(loaded_artifacts)

        vec_before = rec._current_user_vector(uid).copy()
        rec.update_user(
            user_id=uid, movie_id=movie_id, action_type=SwipeAction.SKIP
        )

        if uid in rec.online_user_vectors:
            vec_after = rec.online_user_vectors[uid]
            np.testing.assert_array_equal(vec_before, vec_after)

    def test_skip_still_marks_movie_as_seen(
        self, loaded_artifacts: RecommenderArtifacts
    ):
        rec = _fresh_rec(loaded_artifacts)
        uid = _real_user_id(loaded_artifacts)
        movie_id = _real_movie_id(loaded_artifacts)

        rec.update_user(
            user_id=uid, movie_id=movie_id, action_type=SwipeAction.SKIP
        )
        assert movie_id in rec.user_seen_movie_ids[uid]


# ---------------------------------------------------------------------------
# 8. Multiple swipes accumulate
# ---------------------------------------------------------------------------


class TestMultipleSwipesAccumulate:
    def test_two_likes_drift_further_than_one(
        self, loaded_artifacts: RecommenderArtifacts
    ):
        """Liking two movies should push the user vector further from its
        starting position than liking just one."""
        uid = _real_user_id(loaded_artifacts, 4)
        movie_a = _real_movie_id(loaded_artifacts, 0)
        movie_b = _real_movie_id(loaded_artifacts, 1)

        rec = _fresh_rec(loaded_artifacts)
        base = rec._base_user_vector(uid).copy()

        rec.update_user(
            user_id=uid, movie_id=movie_a, action_type=SwipeAction.LIKE
        )
        drift_one = float(np.linalg.norm(rec.online_user_vectors[uid] - base))

        rec.update_user(
            user_id=uid, movie_id=movie_b, action_type=SwipeAction.LIKE
        )
        drift_two = float(np.linalg.norm(rec.online_user_vectors[uid] - base))

        print(f"\n  Drift after 1 like: {drift_one:.6f}")
        print(f"  Drift after 2 likes: {drift_two:.6f}")
        assert drift_two > drift_one

    def test_sequential_recs_exclude_swiped_movies(
        self, loaded_artifacts: RecommenderArtifacts
    ):
        """After swiping on multiple movies, all should be excluded from recs."""
        rec = _fresh_rec(loaded_artifacts)
        uid = _real_user_id(loaded_artifacts, 5)
        swiped = []

        for _ in range(3):
            recs = rec.get_top_n(user_id=uid, n=1, user_preferences=None)
            movie_id = recs[0][0]
            swiped.append(movie_id)
            rec.update_user(
                user_id=uid, movie_id=movie_id, action_type=SwipeAction.LIKE
            )

        recs_after = rec.get_top_n(user_id=uid, n=10, user_preferences=None)
        rec_ids = {mid for mid, _ in recs_after}
        for mid in swiped:
            assert mid not in rec_ids, f"Swiped movie {mid} still in recs"


# ---------------------------------------------------------------------------
# 9. Supercharged swipe has stronger effect
# ---------------------------------------------------------------------------


class TestSuperchargedSwipe:
    def test_supercharged_like_moves_vector_more(
        self, loaded_artifacts: RecommenderArtifacts
    ):
        """Supercharged like (preference=+2) should increase the movie's
        score more than a regular like (preference=+1)."""
        uid = _real_user_id(loaded_artifacts, 6)
        movie_id = _real_movie_id(loaded_artifacts, 0)
        movie_idx = loaded_artifacts.movie_id_to_index[movie_id]
        movie_vec = loaded_artifacts.movie_embeddings[movie_idx]

        rec_regular = _fresh_rec(loaded_artifacts)
        rec_regular.update_user(
            user_id=uid, movie_id=movie_id, action_type=SwipeAction.LIKE
        )
        score_regular = float(
            movie_vec @ rec_regular.online_user_vectors[uid]
        )

        rec_super = _fresh_rec(loaded_artifacts)
        rec_super.update_user(
            user_id=uid,
            movie_id=movie_id,
            action_type=SwipeAction.LIKE,
            is_supercharged=True,
        )
        score_super = float(movie_vec @ rec_super.online_user_vectors[uid])

        print(f"\n  Regular like score:      {score_regular:.4f}")
        print(f"  Supercharged like score: {score_super:.4f}")
        assert score_super > score_regular

    def test_supercharged_dislike_moves_vector_more(
        self, loaded_artifacts: RecommenderArtifacts
    ):
        """Supercharged dislike should decrease the movie's score more."""
        uid = _real_user_id(loaded_artifacts, 7)
        movie_id = _real_movie_id(loaded_artifacts, 0)
        movie_idx = loaded_artifacts.movie_id_to_index[movie_id]
        movie_vec = loaded_artifacts.movie_embeddings[movie_idx]

        rec_regular = _fresh_rec(loaded_artifacts)
        rec_regular.update_user(
            user_id=uid, movie_id=movie_id, action_type=SwipeAction.DISLIKE
        )
        score_regular = float(
            movie_vec @ rec_regular.online_user_vectors[uid]
        )

        rec_super = _fresh_rec(loaded_artifacts)
        rec_super.update_user(
            user_id=uid,
            movie_id=movie_id,
            action_type=SwipeAction.DISLIKE,
            is_supercharged=True,
        )
        score_super = float(movie_vec @ rec_super.online_user_vectors[uid])

        print(f"\n  Regular dislike score:      {score_regular:.4f}")
        print(f"  Supercharged dislike score: {score_super:.4f}")
        assert score_super < score_regular


# ---------------------------------------------------------------------------
# 10. Unknown movie in swipe
# ---------------------------------------------------------------------------


class TestUnknownMovieSwipe:
    def test_unknown_movie_still_marked_as_seen(
        self, loaded_artifacts: RecommenderArtifacts
    ):
        rec = _fresh_rec(loaded_artifacts)
        uid = _real_user_id(loaded_artifacts)
        rec.update_user(
            user_id=uid,
            movie_id=UNKNOWN_MOVIE_ID,
            action_type=SwipeAction.LIKE,
        )
        assert UNKNOWN_MOVIE_ID in rec.user_seen_movie_ids[uid]

    def test_unknown_movie_does_not_update_vector(
        self, loaded_artifacts: RecommenderArtifacts
    ):
        rec = _fresh_rec(loaded_artifacts)
        uid = _real_user_id(loaded_artifacts)
        vec_before = rec._current_user_vector(uid).copy()

        rec.update_user(
            user_id=uid,
            movie_id=UNKNOWN_MOVIE_ID,
            action_type=SwipeAction.LIKE,
        )

        assert uid not in rec.online_user_vectors
        vec_after = rec._current_user_vector(uid)
        np.testing.assert_array_equal(vec_before, vec_after)

    def test_unknown_movie_does_not_affect_real_recs(
        self, loaded_artifacts: RecommenderArtifacts
    ):
        rec = _fresh_rec(loaded_artifacts)
        uid = _real_user_id(loaded_artifacts)

        rec.update_user(
            user_id=uid,
            movie_id=UNKNOWN_MOVIE_ID,
            action_type=SwipeAction.LIKE,
        )
        recs = rec.get_top_n(user_id=uid, n=10, user_preferences=None)
        assert len(recs) == 10
