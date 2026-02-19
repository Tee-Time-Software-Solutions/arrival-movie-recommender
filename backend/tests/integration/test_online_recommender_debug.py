"""
Verbose debug version of the integration tests.
Run with: pytest tests/integration/test_online_recommender_debug.py -v -s

The -s flag lets you see all the print output.
"""

import numpy as np

from movie_recommender.schemas.interactions import SwipeAction
from movie_recommender.services.recommender.main import Recommender


class TestOnlineRecommenderDebug:

    def test_known_user_gets_personalised_recs(self, recommender: Recommender):
        """User 1 (action fan) should get action movies first."""
        user_vec = recommender._current_user_vector("1")
        scores = recommender.artifacts.movie_embeddings @ user_vec

        print("\n--- KNOWN USER (action fan) ---")
        print(f"  User 1 vector: {user_vec}")
        print(f"  Scores per movie:")
        for mid, idx in recommender.artifacts.movie_id_to_index.items():
            title = recommender.artifacts.movie_id_to_title[mid]
            print(f"    {title} (id={mid}): score = {scores[idx]:.3f}")

        recs = recommender.get_top_n("1", 5, None)
        print(f"  Top-5 recommendations:")
        for i, (mid, title) in enumerate(recs, 1):
            print(f"    {i}. {title} (id={mid})")

        assert {mid for mid, _ in recs[:2]} == {100, 103}

    def test_cold_start_user(self, recommender: Recommender):
        """Unknown user should get recs based on mean of all user vectors."""
        cold_vec = recommender._current_user_vector("999")
        expected_mean = recommender.artifacts.user_embeddings.mean(axis=0)

        print("\n--- COLD START (unknown user 999) ---")
        print(f"  Mean of all user vectors: {expected_mean}")
        print(f"  Cold-start vector:        {cold_vec}")
        print(f"  Vectors match: {np.allclose(cold_vec, expected_mean)}")

        recs = recommender.get_top_n("999", 3, None)
        print(f"  Top-3 recommendations:")
        for i, (mid, title) in enumerate(recs, 1):
            print(f"    {i}. {title} (id={mid})")

        assert recs[0][0] == 103

    def test_like_shifts_vector_toward_movie(self, recommender: Recommender):
        """Liking comedy should add comedy direction to user vector."""
        vec_before = recommender._current_user_vector("1").copy()

        print("\n--- LIKE SHIFTS VECTOR ---")
        print(f"  User 1 before: {vec_before}")

        recommender.update_user("1", 101, SwipeAction.LIKE)
        vec_after = recommender.online_user_vectors["1"]
        delta = vec_after - vec_before

        print(f"  User 1 after LIKE comedy: {vec_after}")
        print(f"  Delta:                    {delta}")
        print(f"  Expected delta:           [0, 0.05, 0, 0]  (eta=0.05 * preference=1 * comedy_vec=[0,1,0,0])")
        print(f"  Comedy component went from {vec_before[1]:.4f} -> {vec_after[1]:.4f}")

        assert vec_after[1] > 0.0

    def test_dislike_shifts_vector_away_from_movie(self, recommender: Recommender):
        """Disliking horror should subtract horror direction from user vector."""
        vec_before = recommender._current_user_vector("1").copy()

        print("\n--- DISLIKE SHIFTS VECTOR ---")
        print(f"  User 1 before: {vec_before}")

        recommender.update_user("1", 104, SwipeAction.DISLIKE)
        vec_after = recommender.online_user_vectors["1"]
        delta = vec_after - vec_before

        print(f"  User 1 after DISLIKE horror: {vec_after}")
        print(f"  Delta:                       {delta}")
        print(f"  Expected delta:              [0, 0, 0, -0.05]  (eta=0.05 * preference=-1 * horror_vec=[0,0,0,1])")
        print(f"  Horror component went from {vec_before[3]:.4f} -> {vec_after[3]:.4f}")

        assert vec_after[3] < 0.0

    def test_seen_movies_excluded(self, recommender: Recommender):
        """After swiping a movie, it should never appear in recs again."""
        recs_before = recommender.get_top_n("1", 5, None)
        ids_before = [mid for mid, _ in recs_before]

        print("\n--- SEEN MOVIE EXCLUSION ---")
        print(f"  Before swipe: {ids_before}")

        recommender.update_user("1", 100, SwipeAction.LIKE)
        recs_after = recommender.get_top_n("1", 5, None)
        ids_after = [mid for mid, _ in recs_after]

        print(f"  Swiped movie 100 (Action Movie)")
        print(f"  After swipe:  {ids_after}")
        print(f"  Movie 100 excluded: {100 not in ids_after}")

        assert 100 not in ids_after

    def test_supercharged_has_double_effect(self, recommender: Recommender):
        """Supercharged like should move the vector 2x as much."""
        # Regular like
        rec_reg = Recommender.__new__(Recommender)
        rec_reg.artifacts = recommender.artifacts
        rec_reg._artifact_load_error = None
        rec_reg.online_user_vectors = {}
        rec_reg.user_seen_movie_ids = {}
        rec_reg.eta = 0.05
        rec_reg.norm_cap = 10.0
        rec_reg.update_user("1", 101, SwipeAction.LIKE, is_supercharged=False)
        vec_reg = rec_reg.online_user_vectors["1"]

        # Supercharged like
        rec_sup = Recommender.__new__(Recommender)
        rec_sup.artifacts = recommender.artifacts
        rec_sup._artifact_load_error = None
        rec_sup.online_user_vectors = {}
        rec_sup.user_seen_movie_ids = {}
        rec_sup.eta = 0.05
        rec_sup.norm_cap = 10.0
        rec_sup.update_user("1", 101, SwipeAction.LIKE, is_supercharged=True)
        vec_sup = rec_sup.online_user_vectors["1"]

        print("\n--- SUPERCHARGED vs REGULAR ---")
        print(f"  Regular LIKE comedy:      {vec_reg}  (comedy={vec_reg[1]:.4f})")
        print(f"  Supercharged LIKE comedy: {vec_sup}  (comedy={vec_sup[1]:.4f})")
        print(f"  Regular delta:      0.05 * 1 * 1.0 = 0.05")
        print(f"  Supercharged delta: 0.05 * 2 * 1.0 = 0.10")
        print(f"  Supercharged effect is 2x: {vec_sup[1] / vec_reg[1]:.1f}x")

        assert vec_sup[1] > vec_reg[1]

    def test_full_browsing_session(self, recommender: Recommender):
        """Simulate a real user session: browse, swipe, see recs change."""
        print("\n--- FULL BROWSING SESSION (User 3, mixed taste) ---")
        print(f"  Starting vector: {recommender._current_user_vector('3')}")

        for step in range(4):
            recs = recommender.get_top_n("3", 2, None)
            if not recs:
                print(f"\n  Step {step + 1}: No more recommendations!")
                break

            movie_id, title = recs[0]
            # Alternate: like, dislike, skip, like
            actions = [SwipeAction.LIKE, SwipeAction.DISLIKE, SwipeAction.SKIP, SwipeAction.LIKE]
            action = actions[step]

            print(f"\n  Step {step + 1}:")
            print(f"    Top recs: {recs}")
            print(f"    Action: {action.value.upper()} on '{title}' (id={movie_id})")

            recommender.update_user("3", movie_id, action)

            vec = recommender._current_user_vector("3")
            seen = recommender.user_seen_movie_ids.get("3", set())
            print(f"    Vector after: {vec}")
            print(f"    Seen movies:  {sorted(seen)}")

        print(f"\n  Session complete.")
        print(f"  Final vector: {recommender._current_user_vector('3')}")
        print(f"  Total seen:   {sorted(recommender.user_seen_movie_ids.get('3', set()))}")

        remaining = recommender.get_top_n("3", 5, None)
        print(f"  Remaining recs: {remaining}")

        assert len(recommender.user_seen_movie_ids.get("3", set())) == 4
