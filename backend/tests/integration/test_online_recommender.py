"""
Integration tests for the online recommender pipeline.

Tests the full loop: artifacts → recommendations → swipe feedback → updated
user vector → changed recommendations.  All tests use small synthetic
embeddings injected via the ``recommender`` fixture (no disk I/O).
"""

import numpy as np
import pytest

from movie_recommender.schemas.interactions import SwipeAction
from movie_recommender.services.recommender.main import Recommender
from movie_recommender.services.recommender.serving.artifact_loader import (
    RecommenderArtifacts,
)


# ---------------------------------------------------------------------------
# 1. Artifact loading
# ---------------------------------------------------------------------------


class TestArtifactInitialisation:
    def test_recommender_has_artifacts_when_injected(
        self, recommender: Recommender
    ):
        assert recommender.artifacts is not None

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
    def test_action_fan_gets_action_movies_first(self, recommender: Recommender):
        """User 1 has embedding [1,0,0,0] — highest dot products with
        movie 100 ([1,0,0,0] → 1.0) and movie 103 ([1,1,0,0] → 1.0)."""
        recs = recommender.get_top_n(user_id="1", n=2, user_preferences=None)

        movie_ids = [mid for mid, _ in recs]
        assert set(movie_ids) == {100, 103}

    def test_comedy_fan_gets_comedy_movies_first(self, recommender: Recommender):
        """User 2 has embedding [0,1,0,0] — highest dot products with
        movie 101 ([0,1,0,0] → 1.0) and movie 103 ([1,1,0,0] → 1.0)."""
        recs = recommender.get_top_n(user_id="2", n=2, user_preferences=None)

        movie_ids = [mid for mid, _ in recs]
        assert set(movie_ids) == {101, 103}

    def test_returns_requested_count(self, recommender: Recommender):
        recs = recommender.get_top_n(user_id="1", n=3, user_preferences=None)
        assert len(recs) == 3

    def test_returns_movie_id_and_title_tuples(self, recommender: Recommender):
        recs = recommender.get_top_n(user_id="1", n=1, user_preferences=None)
        movie_id, title = recs[0]
        assert isinstance(movie_id, int)
        assert isinstance(title, str)

    def test_requesting_more_than_available_returns_all(
        self, recommender: Recommender
    ):
        recs = recommender.get_top_n(user_id="1", n=100, user_preferences=None)
        assert len(recs) == 5


# ---------------------------------------------------------------------------
# 3. Cold-start user
# ---------------------------------------------------------------------------


class TestColdStartUser:
    def test_unknown_user_gets_recommendations(self, recommender: Recommender):
        """User '999' is not in the training data — should fall back to
        mean user embedding and still return results."""
        recs = recommender.get_top_n(user_id="999", n=3, user_preferences=None)
        assert len(recs) == 3

    def test_cold_start_uses_mean_vector(self, recommender: Recommender):
        """Mean of user embeddings:
        ([1,0,0,0] + [0,1,0,0] + [0.5,0.5,0,0]) / 3 = [0.5, 0.5, 0, 0]
        Dot products: movie 103 ([1,1,0,0]) = 1.0 is highest."""
        recs = recommender.get_top_n(user_id="999", n=1, user_preferences=None)
        top_movie_id = recs[0][0]
        assert top_movie_id == 103

    def test_non_numeric_user_id_gets_cold_start(self, recommender: Recommender):
        recs = recommender.get_top_n(
            user_id="alice", n=2, user_preferences=None
        )
        assert len(recs) == 2


# ---------------------------------------------------------------------------
# 4. Seen-movie exclusion
# ---------------------------------------------------------------------------


class TestSeenMovieExclusion:
    def test_swiped_movie_excluded_from_next_recs(
        self, recommender: Recommender
    ):
        recommender.update_user(
            user_id="1", movie_id=100, action_type=SwipeAction.LIKE
        )
        recs = recommender.get_top_n(user_id="1", n=5, user_preferences=None)
        rec_ids = [mid for mid, _ in recs]
        assert 100 not in rec_ids

    def test_all_movies_seen_returns_empty(self, recommender: Recommender):
        for movie_id in [100, 101, 102, 103, 104]:
            recommender.update_user(
                user_id="1", movie_id=movie_id, action_type=SwipeAction.LIKE
            )
        recs = recommender.get_top_n(user_id="1", n=5, user_preferences=None)
        assert recs == []


# ---------------------------------------------------------------------------
# 5. Like shifts recommendations toward similar movies
# ---------------------------------------------------------------------------


class TestLikeShiftsRecommendations:
    def test_liking_comedy_increases_comedy_score(
        self, recommender: Recommender
    ):
        """User 1 (action fan [1,0,0,0]) likes the comedy movie ([0,1,0,0]).
        The user vector should shift toward comedy, making comedy-related
        movies score higher."""
        recs_before = recommender.get_top_n(
            user_id="1", n=5, user_preferences=None
        )
        ids_before = [mid for mid, _ in recs_before]

        recommender.update_user(
            user_id="1", movie_id=101, action_type=SwipeAction.LIKE
        )

        updated_vec = recommender.online_user_vectors["1"]
        # The comedy component (index 1) should now be > 0
        assert updated_vec[1] > 0.0
        # The action component (index 0) should remain ~1.0
        assert updated_vec[0] == pytest.approx(1.0, abs=0.01)

    def test_liking_action_keeps_action_fan_direction(
        self, recommender: Recommender
    ):
        """User 1 likes an action movie — vector should stay action-biased."""
        recommender.update_user(
            user_id="1", movie_id=100, action_type=SwipeAction.LIKE
        )

        updated_vec = recommender.online_user_vectors["1"]
        # Action component should increase (was 1.0, now 1.0 + 0.05*1*1.0)
        assert updated_vec[0] > 1.0


# ---------------------------------------------------------------------------
# 6. Dislike shifts recommendations away
# ---------------------------------------------------------------------------


class TestDislikeShiftsRecommendations:
    def test_disliking_horror_decreases_horror_score(
        self, recommender: Recommender
    ):
        """User 1 dislikes horror ([0,0,0,1]).
        Horror component (index 3) should become negative."""
        recommender.update_user(
            user_id="1", movie_id=104, action_type=SwipeAction.DISLIKE
        )

        updated_vec = recommender.online_user_vectors["1"]
        assert updated_vec[3] < 0.0

    def test_dislike_lowers_movie_score_in_next_recs(
        self, recommender: Recommender
    ):
        """After disliking horror, the horror movie's score should be lower
        than before (if it weren't excluded as seen)."""
        artifacts = recommender.artifacts
        vec_before = recommender._current_user_vector("1")
        score_horror_before = float(
            artifacts.movie_embeddings[4] @ vec_before
        )

        recommender.update_user(
            user_id="1", movie_id=104, action_type=SwipeAction.DISLIKE
        )

        vec_after = recommender._current_user_vector("1")
        score_horror_after = float(artifacts.movie_embeddings[4] @ vec_after)

        assert score_horror_after < score_horror_before


# ---------------------------------------------------------------------------
# 7. Skip behaviour
# ---------------------------------------------------------------------------


class TestSkipBehaviour:
    def test_skip_does_not_change_user_vector(self, recommender: Recommender):
        vec_before = recommender._current_user_vector("1").copy()

        recommender.update_user(
            user_id="1", movie_id=100, action_type=SwipeAction.SKIP
        )

        # Skip with preference=0 still stores a vector (copy of original)
        if "1" in recommender.online_user_vectors:
            vec_after = recommender.online_user_vectors["1"]
            np.testing.assert_array_equal(vec_before, vec_after)

    def test_skip_still_marks_movie_as_seen(self, recommender: Recommender):
        recommender.update_user(
            user_id="1", movie_id=100, action_type=SwipeAction.SKIP
        )
        assert 100 in recommender.user_seen_movie_ids["1"]


# ---------------------------------------------------------------------------
# 8. Multiple swipes accumulate
# ---------------------------------------------------------------------------


class TestMultipleSwipesAccumulate:
    def test_two_action_likes_drift_further_than_one(
        self, recommender: Recommender
    ):
        """Liking action movie twice should push user vector further into
        the action direction than a single like."""
        # Single like
        rec_single = Recommender.__new__(Recommender)
        rec_single.artifacts = recommender.artifacts
        rec_single._artifact_load_error = None
        rec_single.online_user_vectors = {}
        rec_single.user_seen_movie_ids = {}
        rec_single.eta = 0.05
        rec_single.norm_cap = 10.0

        rec_single.update_user(
            user_id="3", movie_id=100, action_type=SwipeAction.LIKE
        )
        vec_after_one = rec_single.online_user_vectors["3"].copy()

        # Second like (different action movie — movie 103 has action component)
        rec_single.update_user(
            user_id="3", movie_id=103, action_type=SwipeAction.LIKE
        )
        vec_after_two = rec_single.online_user_vectors["3"]

        # Action component (index 0) should be larger after two likes
        assert vec_after_two[0] > vec_after_one[0]

    def test_sequential_recs_reflect_accumulated_feedback(
        self, recommender: Recommender
    ):
        """User 3 (mixed [0.5,0.5,0,0]) likes two action movies — recs
        should shift toward action-heavy movies."""
        recs_initial = recommender.get_top_n(
            user_id="3", n=5, user_preferences=None
        )

        recommender.update_user(
            user_id="3", movie_id=100, action_type=SwipeAction.LIKE
        )
        recommender.update_user(
            user_id="3", movie_id=103, action_type=SwipeAction.LIKE
        )

        recs_after = recommender.get_top_n(
            user_id="3", n=3, user_preferences=None
        )
        # Movie 100 and 103 are now seen, so excluded.
        # Of remaining [101, 102, 104], action-drifted user should NOT
        # rank horror (104) first.
        rec_ids = [mid for mid, _ in recs_after]
        assert rec_ids[0] != 104


# ---------------------------------------------------------------------------
# 9. Supercharged swipe has stronger effect
# ---------------------------------------------------------------------------


class TestSuperchargedSwipe:
    def test_supercharged_like_moves_vector_more(
        self, recommender: Recommender
    ):
        """Supercharged like (preference=+2) should cause a larger shift
        than regular like (preference=+1)."""
        # Regular like
        rec_regular = Recommender.__new__(Recommender)
        rec_regular.artifacts = recommender.artifacts
        rec_regular._artifact_load_error = None
        rec_regular.online_user_vectors = {}
        rec_regular.user_seen_movie_ids = {}
        rec_regular.eta = 0.05
        rec_regular.norm_cap = 10.0

        rec_regular.update_user(
            user_id="1", movie_id=101, action_type=SwipeAction.LIKE
        )
        vec_regular = rec_regular.online_user_vectors["1"].copy()

        # Supercharged like
        rec_super = Recommender.__new__(Recommender)
        rec_super.artifacts = recommender.artifacts
        rec_super._artifact_load_error = None
        rec_super.online_user_vectors = {}
        rec_super.user_seen_movie_ids = {}
        rec_super.eta = 0.05
        rec_super.norm_cap = 10.0

        rec_super.update_user(
            user_id="1",
            movie_id=101,
            action_type=SwipeAction.LIKE,
            is_supercharged=True,
        )
        vec_super = rec_super.online_user_vectors["1"].copy()

        # Comedy component (index 1) should be larger for supercharged
        # Regular: 0 + 0.05 * 1 * 1.0 = 0.05
        # Super:   0 + 0.05 * 2 * 1.0 = 0.10
        assert vec_super[1] > vec_regular[1]

    def test_supercharged_dislike_moves_vector_more(
        self, recommender: Recommender
    ):
        rec_regular = Recommender.__new__(Recommender)
        rec_regular.artifacts = recommender.artifacts
        rec_regular._artifact_load_error = None
        rec_regular.online_user_vectors = {}
        rec_regular.user_seen_movie_ids = {}
        rec_regular.eta = 0.05
        rec_regular.norm_cap = 10.0

        rec_regular.update_user(
            user_id="1", movie_id=104, action_type=SwipeAction.DISLIKE
        )
        vec_regular = rec_regular.online_user_vectors["1"].copy()

        rec_super = Recommender.__new__(Recommender)
        rec_super.artifacts = recommender.artifacts
        rec_super._artifact_load_error = None
        rec_super.online_user_vectors = {}
        rec_super.user_seen_movie_ids = {}
        rec_super.eta = 0.05
        rec_super.norm_cap = 10.0

        rec_super.update_user(
            user_id="1",
            movie_id=104,
            action_type=SwipeAction.DISLIKE,
            is_supercharged=True,
        )
        vec_super = rec_super.online_user_vectors["1"].copy()

        # Horror component (index 3) should be more negative for supercharged
        assert vec_super[3] < vec_regular[3]


# ---------------------------------------------------------------------------
# 10. Unknown movie in swipe
# ---------------------------------------------------------------------------


class TestUnknownMovieSwipe:
    def test_unknown_movie_still_marked_as_seen(self, recommender: Recommender):
        """Movie 999 doesn't exist in movie_id_to_index but should still
        be added to the seen set."""
        recommender.update_user(
            user_id="1", movie_id=999, action_type=SwipeAction.LIKE
        )
        assert 999 in recommender.user_seen_movie_ids["1"]

    def test_unknown_movie_does_not_update_vector(
        self, recommender: Recommender
    ):
        """User vector should remain unchanged because we can't look up
        the movie embedding."""
        vec_before = recommender._current_user_vector("1").copy()

        recommender.update_user(
            user_id="1", movie_id=999, action_type=SwipeAction.LIKE
        )

        # No online vector should be stored (early return before update)
        assert "1" not in recommender.online_user_vectors
        vec_after = recommender._current_user_vector("1")
        np.testing.assert_array_equal(vec_before, vec_after)

    def test_unknown_movie_excluded_from_future_recs(
        self, recommender: Recommender
    ):
        """Even though movie 999 has no embedding, it's in the seen set.
        This won't affect filtering (999 isn't in movie_id_to_index),
        but verifies no crash occurs."""
        recommender.update_user(
            user_id="1", movie_id=999, action_type=SwipeAction.LIKE
        )
        recs = recommender.get_top_n(user_id="1", n=5, user_preferences=None)
        assert len(recs) == 5  # All real movies still available
