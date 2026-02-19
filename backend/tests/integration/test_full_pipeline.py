"""
Full offline-to-online pipeline integration test using real MovieLens 20M data.

Downloads the MovieLens 20M dataset (~200MB) if not present, runs the entire
offline pipeline (preprocess -> filter -> split -> build matrix -> train ALS ->
evaluate), then verifies the produced artifacts can be loaded by the online
recommender and used for recommendations + swipe feedback.

First run: ~5 min (download + pipeline). Subsequent runs: instant (cached).

Run with:  pytest tests/integration/test_full_pipeline.py -v -s
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from movie_recommender.schemas.interactions import SwipeAction
from movie_recommender.services.recommender.main import Recommender
from movie_recommender.services.recommender.serving.artifact_loader import (
    RecommenderArtifacts,
)

# ---------------------------------------------------------------------------
# Vector visualization helpers
# ---------------------------------------------------------------------------

_BLOCKS = " ▁▂▃▄▅▆▇█"


def _sparkline(vec: np.ndarray, vmin: float = None, vmax: float = None) -> str:
    """Render a vector as a one-line sparkline (one char per dimension)."""
    if vmin is None:
        vmin = float(vec.min())
    if vmax is None:
        vmax = float(vec.max())
    if vmax <= vmin:
        return "▄" * len(vec)
    result = []
    for v in vec:
        normalized = (float(v) - vmin) / (vmax - vmin)
        normalized = max(0.0, min(1.0, normalized))
        result.append(_BLOCKS[int(normalized * 8)])
    return "".join(result)


def _print_vector_comparison(
    vec_before: np.ndarray,
    vec_after: np.ndarray,
    movie_vec: np.ndarray,
    eta: float,
    preference: int,
    norm_cap: float,
    top_n: int = 8,
):
    """Print a clean before/after comparison with sparklines and top-dims table."""
    # Use shared scale for before/after sparklines so they're visually comparable
    shared_min = min(float(vec_before.min()), float(vec_after.min()))
    shared_max = max(float(vec_before.max()), float(vec_after.max()))

    delta = vec_after - vec_before
    raw_nudge = eta * preference * movie_vec
    nudge_norm = float(np.linalg.norm(raw_nudge))

    print(f"  Before  norm={np.linalg.norm(vec_before):<8.4f} {_sparkline(vec_before, shared_min, shared_max)}")
    print(f"  Movie   norm={np.linalg.norm(movie_vec):<8.4f} {_sparkline(movie_vec)}")
    print(f"  After   norm={np.linalg.norm(vec_after):<8.4f} {_sparkline(vec_after, shared_min, shared_max)}")
    print()

    # Show the math
    capped = np.linalg.norm(vec_after) < np.linalg.norm(vec_before + raw_nudge) - 0.001
    print(f"  Update: user + {eta} * {preference:+d} * movie  (nudge norm={nudge_norm:.6f})")
    if capped:
        raw_norm = float(np.linalg.norm(vec_before + raw_nudge))
        print(f"  Norm cap applied: {raw_norm:.4f} -> {norm_cap:.1f}  (rescaled entire vector)")
    print(f"  Actual delta norm: {np.linalg.norm(delta):.6f}")
    print()

    # Top-N changed dimensions table
    top_dims = np.argsort(-np.abs(delta))[:top_n]
    print(f"  {'dim':>5s}   {'before':>8s}   {'after':>8s}   {'delta':>8s}   {'movie':>8s}")
    print(f"  {'───':>5s}   {'──────':>8s}   {'─────':>8s}   {'─────':>8s}   {'─────':>8s}")
    for d in top_dims:
        print(f"  {d:5d}   {vec_before[d]:+8.4f}   {vec_after[d]:+8.4f}   {delta[d]:+8.4f}   {movie_vec[d]:+8.4f}")


# ---------------------------------------------------------------------------
# 1. Offline pipeline execution
# ---------------------------------------------------------------------------


class TestOfflinePipelineRuns:
    def test_pipeline_completes(self, pipeline_dir: Path):
        """The full 8-step pipeline ran without raising."""
        print(f"\n--- OFFLINE PIPELINE ---")
        print(f"  Project root: {pipeline_dir}")
        print(f"  Dataset: MovieLens 20M")
        print(f"  Pipeline completed: Yes (no exceptions)")
        assert pipeline_dir.exists()

    def test_intermediate_files_created(self, pipeline_dir: Path):
        """All intermediate parquet files were produced."""
        expected = [
            "data/processed/movies_clean.parquet",
            "data/processed/interactions_clean.parquet",
            "data/processed/interactions_filtered.parquet",
            "data/processed/movies_filtered.parquet",
            "data/splits/train.parquet",
            "data/splits/val.parquet",
            "data/splits/test.parquet",
        ]
        print("\n--- INTERMEDIATE FILES ---")
        all_exist = True
        for rel in expected:
            exists = (pipeline_dir / rel).exists()
            status = "FOUND" if exists else "MISSING"
            print(f"  {rel}: {status}")
            if not exists:
                all_exist = False
            assert exists, f"Missing: {rel}"
        print(f"  All {len(expected)} files present: {all_exist}")

    def test_artifact_files_created(self, pipeline_dir: Path):
        """All final artifact files were produced."""
        expected = [
            "artifacts/movie_embeddings.npy",
            "artifacts/user_embeddings.npy",
            "artifacts/mappings.json",
            "artifacts/R_train.npz",
            "artifacts/model_info.json",
        ]
        print("\n--- ARTIFACT FILES ---")
        for rel in expected:
            exists = (pipeline_dir / rel).exists()
            size = (pipeline_dir / rel).stat().st_size if exists else 0
            status = f"FOUND ({size:,} bytes)" if exists else "MISSING"
            print(f"  {rel}: {status}")
            assert exists, f"Missing: {rel}"


# ---------------------------------------------------------------------------
# 2. Data survives filtering
# ---------------------------------------------------------------------------


class TestFilteringSurvival:
    def test_interactions_not_empty_after_filtering(self, pipeline_dir: Path):
        df = pd.read_parquet(
            pipeline_dir / "data/processed/interactions_filtered.parquet"
        )
        print(f"\n--- FILTERING SURVIVAL ---")
        print(f"  Interactions after filtering: {len(df):,}")
        assert len(df) > 100_000, f"Expected >100k interactions, got {len(df):,}"

    def test_users_survived(self, pipeline_dir: Path):
        """With MovieLens 20M, many users should survive filtering."""
        df = pd.read_parquet(
            pipeline_dir / "data/processed/interactions_filtered.parquet"
        )
        n_users = df["user_id"].nunique()
        print(f"\n--- USER SURVIVAL ---")
        print(f"  Users after filtering: {n_users:,}")
        assert n_users > 5_000, f"Expected >5000 users, got {n_users:,}"

    def test_movies_survived(self, pipeline_dir: Path):
        """With MovieLens 20M, many movies should survive filtering."""
        df = pd.read_parquet(
            pipeline_dir / "data/processed/interactions_filtered.parquet"
        )
        n_movies = df["movie_id"].nunique()
        print(f"\n--- MOVIE SURVIVAL ---")
        print(f"  Movies after filtering: {n_movies:,}")
        assert n_movies > 5_000, f"Expected >5000 movies, got {n_movies:,}"

    def test_split_has_train_val_test(self, pipeline_dir: Path):
        train = pd.read_parquet(pipeline_dir / "data/splits/train.parquet")
        val = pd.read_parquet(pipeline_dir / "data/splits/val.parquet")
        test = pd.read_parquet(pipeline_dir / "data/splits/test.parquet")
        filtered = pd.read_parquet(
            pipeline_dir / "data/processed/interactions_filtered.parquet"
        )
        total = len(train) + len(val) + len(test)

        print(f"\n--- CHRONOLOGICAL SPLIT ---")
        print(f"  Train: {len(train):,} ({len(train)/total*100:.0f}%)")
        print(f"  Val:   {len(val):,} ({len(val)/total*100:.0f}%)")
        print(f"  Test:  {len(test):,} ({len(test)/total*100:.0f}%)")
        print(f"  Total: {total:,} (filtered: {len(filtered):,})")
        print(f"  Adds up: {total == len(filtered)}")

        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert total == len(filtered)


# ---------------------------------------------------------------------------
# 3. Artifact shapes and consistency
# ---------------------------------------------------------------------------


class TestArtifactShapes:
    def test_embedding_dimensions_match(self, pipeline_dir: Path):
        """Movie and user embeddings should have the same number of columns (64)."""
        movies = np.load(pipeline_dir / "artifacts/movie_embeddings.npy")
        users = np.load(pipeline_dir / "artifacts/user_embeddings.npy")

        print(f"\n--- EMBEDDING DIMENSIONS ---")
        print(f"  Movie embeddings: {movies.shape} (dtype={movies.dtype})")
        print(f"  User embeddings:  {users.shape} (dtype={users.dtype})")
        print(f"  Dims match: {movies.shape[1]} == {users.shape[1]} == 64")

        assert movies.shape[1] == users.shape[1] == 64

    def test_embedding_row_counts_match_mappings(self, pipeline_dir: Path):
        movies = np.load(pipeline_dir / "artifacts/movie_embeddings.npy")
        users = np.load(pipeline_dir / "artifacts/user_embeddings.npy")

        with open(pipeline_dir / "artifacts/mappings.json") as f:
            mappings = json.load(f)

        n_movie_map = len(mappings["movie_id_to_index"])
        n_user_map = len(mappings["user_id_to_index"])

        print(f"\n--- EMBEDDING vs MAPPING COUNTS ---")
        print(f"  Movie embeddings rows: {movies.shape[0]}, mapping entries: {n_movie_map}, match: {movies.shape[0] == n_movie_map}")
        print(f"  User embeddings rows:  {users.shape[0]}, mapping entries: {n_user_map}, match: {users.shape[0] == n_user_map}")

        assert movies.shape[0] == n_movie_map
        assert users.shape[0] == n_user_map

    def test_index_to_movie_id_is_inverse(self, pipeline_dir: Path):
        with open(pipeline_dir / "artifacts/mappings.json") as f:
            mappings = json.load(f)

        mid_to_idx = {int(k): int(v) for k, v in mappings["movie_id_to_index"].items()}
        idx_to_mid = {int(k): int(v) for k, v in mappings["index_to_movie_id"].items()}

        mismatches = 0
        for mid, idx in mid_to_idx.items():
            if idx_to_mid.get(idx) != mid:
                mismatches += 1

        print(f"\n--- MAPPING INVERSE CHECK ---")
        print(f"  movie_id_to_index entries: {len(mid_to_idx)}")
        print(f"  index_to_movie_id entries: {len(idx_to_mid)}")
        print(f"  Mismatches: {mismatches}")
        print(f"  Bijective: {mismatches == 0}")

        for mid, idx in mid_to_idx.items():
            assert idx_to_mid[idx] == mid

    def test_all_indices_in_bounds(self, pipeline_dir: Path):
        movies = np.load(pipeline_dir / "artifacts/movie_embeddings.npy")
        users = np.load(pipeline_dir / "artifacts/user_embeddings.npy")

        with open(pipeline_dir / "artifacts/mappings.json") as f:
            mappings = json.load(f)

        movie_oob = sum(1 for v in mappings["movie_id_to_index"].values() if int(v) >= movies.shape[0])
        user_oob = sum(1 for v in mappings["user_id_to_index"].values() if int(v) >= users.shape[0])

        print(f"\n--- INDEX BOUNDS CHECK ---")
        print(f"  Movie indices: 0..{movies.shape[0]-1}, out-of-bounds: {movie_oob}")
        print(f"  User indices:  0..{users.shape[0]-1}, out-of-bounds: {user_oob}")

        assert movie_oob == 0
        assert user_oob == 0

    def test_model_info_consistent(self, pipeline_dir: Path):
        with open(pipeline_dir / "artifacts/model_info.json") as f:
            info = json.load(f)

        movies = np.load(pipeline_dir / "artifacts/movie_embeddings.npy")
        users = np.load(pipeline_dir / "artifacts/user_embeddings.npy")

        print(f"\n--- MODEL INFO ---")
        print(f"  factors:       {info['factors']} (expected 64)")
        print(f"  embedding_dim: {info['embedding_dim']} (expected 64)")
        print(f"  num_movies:    {info['num_movies']} (actual: {movies.shape[0]})")
        print(f"  num_users:     {info['num_users']} (actual: {users.shape[0]})")
        print(f"  regularization: {info['regularization']}")
        print(f"  iterations:    {info['iterations']}")
        print(f"  alpha:         {info['alpha']}")

        assert info["factors"] == 64
        assert info["num_movies"] == movies.shape[0]
        assert info["num_users"] == users.shape[0]
        assert info["embedding_dim"] == 64


# ---------------------------------------------------------------------------
# 4. Artifact loader (real loading path)
# ---------------------------------------------------------------------------


class TestArtifactLoader:
    def test_load_returns_artifacts_dataclass(self, loaded_artifacts):
        print(f"\n--- ARTIFACT LOADER ---")
        print(f"  Returned type: {type(loaded_artifacts).__name__}")
        print(f"  Is RecommenderArtifacts: {isinstance(loaded_artifacts, RecommenderArtifacts)}")
        assert isinstance(loaded_artifacts, RecommenderArtifacts)

    def test_loaded_embeddings_are_numpy(self, loaded_artifacts):
        print(f"\n--- LOADED EMBEDDING TYPES ---")
        print(f"  movie_embeddings: {type(loaded_artifacts.movie_embeddings).__name__} {loaded_artifacts.movie_embeddings.shape}")
        print(f"  user_embeddings:  {type(loaded_artifacts.user_embeddings).__name__} {loaded_artifacts.user_embeddings.shape}")
        assert isinstance(loaded_artifacts.movie_embeddings, np.ndarray)
        assert isinstance(loaded_artifacts.user_embeddings, np.ndarray)

    def test_loaded_mappings_are_populated(self, loaded_artifacts):
        print(f"\n--- LOADED MAPPING SIZES ---")
        print(f"  user_id_to_index:  {len(loaded_artifacts.user_id_to_index)} entries")
        print(f"  movie_id_to_index: {len(loaded_artifacts.movie_id_to_index)} entries")
        print(f"  index_to_movie_id: {len(loaded_artifacts.index_to_movie_id)} entries")
        print(f"  movie_id_to_title: {len(loaded_artifacts.movie_id_to_title)} entries")
        assert len(loaded_artifacts.user_id_to_index) > 0
        assert len(loaded_artifacts.movie_id_to_index) > 0
        assert len(loaded_artifacts.index_to_movie_id) > 0
        assert len(loaded_artifacts.movie_id_to_title) > 0

    def test_every_movie_has_a_title(self, loaded_artifacts):
        missing = [
            mid for mid in loaded_artifacts.movie_id_to_index
            if mid not in loaded_artifacts.movie_id_to_title
        ]
        print(f"\n--- MOVIE TITLE COVERAGE ---")
        print(f"  Movies in index: {len(loaded_artifacts.movie_id_to_index)}")
        print(f"  Movies with titles: {len(loaded_artifacts.movie_id_to_title)}")
        print(f"  Missing titles: {len(missing)}")
        if not missing:
            sample = list(loaded_artifacts.movie_id_to_title.items())[:3]
            for mid, title in sample:
                print(f"    id={mid}: \"{title}\"")
        assert len(missing) == 0


# ---------------------------------------------------------------------------
# 5. Online recommender with real artifacts
# ---------------------------------------------------------------------------


class TestOnlineWithRealArtifacts:
    def test_known_user_gets_recommendations(
        self, pipeline_recommender: Recommender, loaded_artifacts: RecommenderArtifacts
    ):
        """A user that exists in training data gets results."""
        # Pick a real user from the mappings
        first_user_id = str(next(iter(loaded_artifacts.user_id_to_index)))
        recs = pipeline_recommender.get_top_n(user_id=first_user_id, n=5, user_preferences=None)
        print(f"\n--- KNOWN USER RECOMMENDATIONS ---")
        print(f"  User: {first_user_id} (in training data)")
        print(f"  Requested: 5, Returned: {len(recs)}")
        for i, (mid, title) in enumerate(recs, 1):
            print(f"    {i}. {title} (id={mid})")
        assert len(recs) > 0
        assert len(recs) <= 5

    def test_cold_start_user_gets_recommendations(
        self, pipeline_recommender: Recommender
    ):
        """An unknown user falls back to mean embedding and still gets results."""
        recs = pipeline_recommender.get_top_n(
            user_id="99999999", n=5, user_preferences=None
        )
        print(f"\n--- COLD-START USER RECOMMENDATIONS ---")
        print(f"  User: 99999999 (NOT in training data)")
        print(f"  Fallback: mean of all user embeddings")
        print(f"  Returned: {len(recs)} recommendations")
        for i, (mid, title) in enumerate(recs, 1):
            print(f"    {i}. {title} (id={mid})")
        assert len(recs) > 0

    def test_recommendation_ids_exist_in_mappings(
        self, pipeline_recommender: Recommender, loaded_artifacts: RecommenderArtifacts
    ):
        first_user_id = str(next(iter(loaded_artifacts.user_id_to_index)))
        recs = pipeline_recommender.get_top_n(user_id=first_user_id, n=5, user_preferences=None)
        print(f"\n--- REC ID VALIDITY ---")
        all_valid = True
        for mid, title in recs:
            in_index = mid in pipeline_recommender.artifacts.movie_id_to_index
            in_title = mid in pipeline_recommender.artifacts.movie_id_to_title
            valid = in_index and in_title
            print(f"  id={mid} \"{title}\": in_index={in_index}, has_title={in_title}")
            if not valid:
                all_valid = False
            assert in_index
            assert in_title
        print(f"  All valid: {all_valid}")

    def test_different_users_get_different_top_pick(
        self, pipeline_recommender: Recommender, loaded_artifacts: RecommenderArtifacts
    ):
        """With trained ALS embeddings, different users should generally have
        different preferences.  Check that not all users get the same #1 movie."""
        print(f"\n--- PERSONALISATION CHECK ---")
        # Sample up to 20 real users from the mappings
        user_ids = list(loaded_artifacts.user_id_to_index.keys())[:20]
        top_picks = {}
        for uid in user_ids:
            recs = pipeline_recommender.get_top_n(
                user_id=str(uid), n=1, user_preferences=None
            )
            if recs:
                mid, title = recs[0]
                top_picks[uid] = (mid, title)
                print(f"  User {uid} top pick: {title} (id={mid})")
        distinct = len(set(mid for mid, _ in top_picks.values()))
        print(f"  Distinct top picks across {len(top_picks)} users: {distinct}")
        assert distinct >= 2


# ---------------------------------------------------------------------------
# 6. Full loop: offline -> online -> swipe -> changed recs
# ---------------------------------------------------------------------------


class TestFullLoop:
    def _get_real_user_id(self, loaded_artifacts: RecommenderArtifacts, index: int = 0) -> str:
        """Get a real user ID from the mappings."""
        user_ids = list(loaded_artifacts.user_id_to_index.keys())
        return str(user_ids[index])

    def test_swipe_excludes_movie_from_future_recs(
        self, pipeline_recommender: Recommender, loaded_artifacts: RecommenderArtifacts
    ):
        user_id = self._get_real_user_id(loaded_artifacts, 0)
        vec_before = pipeline_recommender._current_user_vector(user_id).copy()

        recs = pipeline_recommender.get_top_n(user_id=user_id, n=5, user_preferences=None)
        top_movie_id, top_title = recs[0]
        ids_before = [mid for mid, _ in recs]
        movie_idx = loaded_artifacts.movie_id_to_index[top_movie_id]
        movie_vec = loaded_artifacts.movie_embeddings[movie_idx]

        pipeline_recommender.update_user(
            user_id=user_id, movie_id=top_movie_id, action_type=SwipeAction.LIKE
        )
        vec_after = pipeline_recommender._current_user_vector(user_id)

        recs_after = pipeline_recommender.get_top_n(
            user_id=user_id, n=5, user_preferences=None
        )
        ids_after = [mid for mid, _ in recs_after]

        print(f"\n--- SEEN-MOVIE EXCLUSION (User {user_id}) ---")
        print(f"  LIKE on \"{top_title}\" (id={top_movie_id})")
        _print_vector_comparison(
            vec_before, vec_after, movie_vec,
            eta=pipeline_recommender.eta, preference=1,
            norm_cap=pipeline_recommender.norm_cap,
        )
        print(f"  Recs before: {ids_before}")
        print(f"  Recs after:  {ids_after}")
        print(f"  Excluded: {top_movie_id not in ids_after}")

        assert top_movie_id not in ids_after

    def test_like_updates_user_vector(
        self, pipeline_recommender: Recommender, loaded_artifacts: RecommenderArtifacts
    ):
        user_id = self._get_real_user_id(loaded_artifacts, 1)
        vec_before = pipeline_recommender._current_user_vector(user_id).copy()

        first_rec = pipeline_recommender.get_top_n(
            user_id=user_id, n=1, user_preferences=None
        )
        movie_id, title = first_rec[0]
        movie_idx = loaded_artifacts.movie_id_to_index[movie_id]
        movie_vec = loaded_artifacts.movie_embeddings[movie_idx]

        pipeline_recommender.update_user(
            user_id=user_id, movie_id=movie_id, action_type=SwipeAction.LIKE
        )

        vec_after = pipeline_recommender._current_user_vector(user_id)

        print(f"\n--- LIKE UPDATES VECTOR (User {user_id}) ---")
        print(f"  LIKE on \"{title}\" (id={movie_id})")
        _print_vector_comparison(
            vec_before, vec_after, movie_vec,
            eta=pipeline_recommender.eta, preference=1,
            norm_cap=pipeline_recommender.norm_cap,
        )

        assert not np.array_equal(vec_before, vec_after), "Vector should have changed"

    def test_dislike_moves_vector_away(
        self, pipeline_recommender: Recommender, loaded_artifacts: RecommenderArtifacts
    ):
        """Disliking a movie should decrease its score for the user."""
        user_id = self._get_real_user_id(loaded_artifacts, 2)
        artifacts = pipeline_recommender.artifacts
        recs = pipeline_recommender.get_top_n(user_id=user_id, n=5, user_preferences=None)
        movie_id, title = recs[-1]  # pick last (lowest-scored)
        movie_idx = artifacts.movie_id_to_index[movie_id]
        movie_vec = artifacts.movie_embeddings[movie_idx]

        vec_before = pipeline_recommender._current_user_vector(user_id)
        score_before = float(movie_vec @ vec_before)

        pipeline_recommender.update_user(
            user_id=user_id, movie_id=movie_id, action_type=SwipeAction.DISLIKE
        )

        vec_after = pipeline_recommender._current_user_vector(user_id)
        score_after = float(movie_vec @ vec_after)

        print(f"\n--- DISLIKE LOWERS SCORE (User {user_id}) ---")
        print(f"  DISLIKE on \"{title}\" (id={movie_id})")
        _print_vector_comparison(
            vec_before, vec_after, movie_vec,
            eta=pipeline_recommender.eta, preference=-1,
            norm_cap=pipeline_recommender.norm_cap,
        )
        print(f"  Score: {score_before:.4f} -> {score_after:.4f}  ({score_after - score_before:+.4f})")

        assert score_after < score_before

    def test_full_session_multiple_swipes(
        self, pipeline_recommender: Recommender, loaded_artifacts: RecommenderArtifacts
    ):
        """Simulate a browsing session: get recs, swipe, repeat.
        After several swipes the seen set grows and vector changes."""
        user_id = self._get_real_user_id(loaded_artifacts, 3)
        artifacts = pipeline_recommender.artifacts
        actions = [
            SwipeAction.LIKE,
            SwipeAction.DISLIKE,
            SwipeAction.SKIP,
            SwipeAction.LIKE,
            SwipeAction.DISLIKE,
        ]
        preference_map = {SwipeAction.LIKE: 1, SwipeAction.DISLIKE: -1, SwipeAction.SKIP: 0}

        vec_start = pipeline_recommender._current_user_vector(user_id).copy()
        print(f"\n--- FULL BROWSING SESSION (User {user_id}) ---")
        print(f"  eta={pipeline_recommender.eta}, norm_cap={pipeline_recommender.norm_cap}")
        print(f"  Start  norm={np.linalg.norm(vec_start):<8.4f} {_sparkline(vec_start)}")
        print()

        for step in range(5):
            recs = pipeline_recommender.get_top_n(
                user_id=user_id, n=3, user_preferences=None
            )
            if not recs:
                print(f"  Step {step+1}: No more recommendations!")
                break

            movie_id, title = recs[0]
            action = actions[step]
            movie_idx = artifacts.movie_id_to_index[movie_id]
            movie_vec = artifacts.movie_embeddings[movie_idx]

            vec_pre = pipeline_recommender._current_user_vector(user_id).copy()
            pipeline_recommender.update_user(
                user_id=user_id, movie_id=movie_id, action_type=action
            )
            vec_post = pipeline_recommender._current_user_vector(user_id)
            step_delta = vec_post - vec_pre
            pref = preference_map[action]

            seen = pipeline_recommender.user_seen_movie_ids.get(user_id, set())
            print(f"  Step {step+1}: {action.value.upper():7s} \"{title}\" (pref={pref:+d})")
            print(f"    movie  norm={np.linalg.norm(movie_vec):<8.4f} {_sparkline(movie_vec)}")
            print(f"    user   norm={np.linalg.norm(vec_post):<8.4f} {_sparkline(vec_post, float(vec_start.min()), float(vec_start.max()))}  delta_norm={np.linalg.norm(step_delta):.6f}")

        vec_end = pipeline_recommender._current_user_vector(user_id)
        total_drift = vec_end - vec_start
        print()
        print(f"  Start  norm={np.linalg.norm(vec_start):<8.4f} {_sparkline(vec_start)}")
        print(f"  Final  norm={np.linalg.norm(vec_end):<8.4f} {_sparkline(vec_end, float(vec_start.min()), float(vec_start.max()))}")
        print(f"  Total drift norm: {np.linalg.norm(total_drift):.6f}")

        # Show top dims that drifted over the full session
        top_dims = np.argsort(-np.abs(total_drift))[:6]
        print(f"\n  {'dim':>5s}   {'start':>8s}   {'final':>8s}   {'drift':>8s}")
        print(f"  {'───':>5s}   {'─────':>8s}   {'─────':>8s}   {'─────':>8s}")
        for d in top_dims:
            print(f"  {d:5d}   {vec_start[d]:+8.4f}   {vec_end[d]:+8.4f}   {total_drift[d]:+8.4f}")

        seen = pipeline_recommender.user_seen_movie_ids.get(user_id, set())
        remaining = pipeline_recommender.get_top_n(
            user_id=user_id, n=10, user_preferences=None
        )

        print(f"\n  Total seen: {len(seen)}, Remaining recs: {len(remaining)}, Online vector stored: {user_id in pipeline_recommender.online_user_vectors}")

        assert len(seen) == 5, f"Expected 5 seen movies, got {len(seen)}"
        assert user_id in pipeline_recommender.online_user_vectors


# ---------------------------------------------------------------------------
# 7. Single-user journey: one user through the entire recommendation lifecycle
# ---------------------------------------------------------------------------


class TestSingleUserJourney:
    """Follow a cold-start user through a recommendation session using
    real ALS-trained 64-dim embeddings from MovieLens 20M.

    Picks recognizable movies across genres to swipe on, and tracks
    "probe" movies (not swiped) to observe how scores shift by genre.

    Uses eta=3.0 to compensate for the embedding scale mismatch
    (movie norms ~0.17 vs user norms ~10). TestFullLoop shows the
    default eta=0.05 behavior.
    """

    @staticmethod
    def _find_movie(artifacts, search_term):
        """Find a movie by title substring. Returns (movie_id, title)."""
        for mid, title in artifacts.movie_id_to_title.items():
            if search_term.lower() in title.lower() and mid in artifacts.movie_id_to_index:
                return mid, title
        return None, None

    def test_user_recommendation_journey(
        self, loaded_artifacts: RecommenderArtifacts, pipeline_dir: Path
    ):
        artifacts = loaded_artifacts
        find = lambda s: self._find_movie(artifacts, s)

        # Fresh recommender with eta high enough to see real effects
        rec = Recommender.__new__(Recommender)
        rec.artifacts = artifacts
        rec._artifact_load_error = None
        rec.online_user_vectors = {}
        rec.user_seen_movie_ids = {}
        rec.eta = 3.0
        rec.norm_cap = 20.0

        USER = "999999"  # cold-start user (will use mean embedding)

        # -- Movies to swipe on --
        swipe_plan = [
            (SwipeAction.LIKE,    "Die Hard"),
            (SwipeAction.LIKE,    "Terminator 2"),
            (SwipeAction.LIKE,    "Aliens"),
            (SwipeAction.DISLIKE, "Sleepless in Seattle"),
            (SwipeAction.SKIP,    "Forrest Gump"),
            (SwipeAction.LIKE,    "Matrix, The"),
        ]

        # -- Probe movies to track scores (NOT swiped, just observed) --
        probe_def = {
            "Action":  ["Predator", "Total Recall", "RoboCop"],
            "Romance": ["Pretty Woman", "Notting Hill", "Four Weddings"],
            "Horror":  ["Shining, The", "Exorcist, The", "Nightmare on Elm St"],
            "Drama":   ["Schindler's List", "Good Will Hunting", "American Beauty"],
        }

        # Resolve probe movies
        probes = {}
        for genre, searches in probe_def.items():
            found = []
            for s in searches:
                mid, title = find(s)
                if mid is not None:
                    found.append((mid, title))
            probes[genre] = found

        # -- Helpers --

        def score_probes():
            vec = rec._current_user_vector(USER)
            return {
                (genre, mid): float(artifacts.movie_embeddings[artifacts.movie_id_to_index[mid]] @ vec)
                for genre, movies in probes.items()
                for mid, _ in movies
            }

        def avg_by_genre(scores):
            avgs = {}
            for genre, movies in probes.items():
                vals = [scores[(genre, mid)] for mid, _ in movies]
                avgs[genre] = sum(vals) / len(vals) if vals else 0.0
            return avgs

        def print_probe_table(scores, prev=None):
            avgs = avg_by_genre(scores)
            prev_avgs = avg_by_genre(prev) if prev else None
            for genre in probe_def:
                if prev_avgs:
                    d = avgs[genre] - prev_avgs[genre]
                    arrow = "^" if d > 0.0005 else ("v" if d < -0.0005 else " ")
                    print(f"    {genre:<8s}  avg {avgs[genre]:+.4f}  {arrow} {d:+.4f}")
                else:
                    print(f"    {genre:<8s}  avg {avgs[genre]:+.4f}")
                for mid, title in probes[genre]:
                    s = scores[(genre, mid)]
                    if prev:
                        d = s - prev[(genre, mid)]
                        print(f"      {title:<38s} {s:+.4f}  ({d:+.4f})")
                    else:
                        print(f"      {title:<38s} {s:+.4f}")

        # ===============================================================

        print(f"\n{'='*75}")
        print(f"  COLD-START USER JOURNEY  (real 64-dim ALS embeddings)")
        print(f"  eta={rec.eta}, norm_cap={rec.norm_cap}")
        n_users = len(artifacts.user_id_to_index)
        print(f"  Starting from mean of {n_users:,} user embeddings")
        print(f"{'='*75}")

        # -- Print resolved movies --
        print(f"\n  Swipe targets:")
        for action, search in swipe_plan:
            mid, title = find(search)
            assert mid is not None, f"Could not find movie matching '{search}'"
            print(f"    {action.value.upper():7s}  {title}")

        print(f"\n  Probe movies (scores tracked, never swiped):")
        for genre in probe_def:
            names = [title for _, title in probes[genre]]
            print(f"    {genre:<8s}  {', '.join(names)}")

        # -- Initial state --
        scores = score_probes()
        vec = rec._current_user_vector(USER)
        spark_min, spark_max = float(vec.min()), float(vec.max())

        print(f"\n  START  norm={np.linalg.norm(vec):.4f}")
        print(f"  {_sparkline(vec)}")
        print_probe_table(scores)

        initial_scores = scores
        initial_avgs = avg_by_genre(scores)

        # -- Swipe steps --
        scores_before_skip = None
        scores_after_skip = None

        for step, (action, search) in enumerate(swipe_plan, 1):
            mid, title = find(search)
            prev = scores

            vec_before = rec._current_user_vector(USER).copy()
            rec.update_user(user_id=USER, movie_id=mid, action_type=action)
            vec_after = rec._current_user_vector(USER)

            scores = score_probes()
            delta_norm = float(np.linalg.norm(vec_after - vec_before))

            # Capture skip step for assertion
            if action == SwipeAction.SKIP:
                scores_before_skip = prev
                scores_after_skip = scores

            label = action.value.upper()
            print(f"\n  STEP {step}  {label} \"{title}\"  norm={np.linalg.norm(vec_after):.4f}  delta={delta_norm:.4f}")
            print(f"  {_sparkline(vec_after, spark_min, spark_max)}")
            print_probe_table(scores, prev)

        # -- Summary --
        vec_final = rec._current_user_vector(USER)
        vec_initial = artifacts.user_embeddings.mean(axis=0)
        drift = float(np.linalg.norm(vec_final - vec_initial))
        seen = rec.user_seen_movie_ids.get(USER, set())
        final_avgs = avg_by_genre(scores)

        print(f"\n  {'='*71}")
        print(f"  JOURNEY SUMMARY")
        print(f"  {'='*71}")
        print(f"  Swipes: 4 LIKES (action/sci-fi), 1 DISLIKE (romance), 1 SKIP (drama)")
        print(f"  Total seen: {len(seen)},  Vector drift: {drift:.4f}")
        print()
        print(f"  Cumulative score shift by genre:")
        for genre in probe_def:
            d = final_avgs[genre] - initial_avgs[genre]
            bar_unit = max(abs(final_avgs[g] - initial_avgs[g]) for g in probe_def)
            if bar_unit > 0:
                bar_len = int(abs(d) / bar_unit * 25)
            else:
                bar_len = 0
            if d >= 0:
                bar = "+" * bar_len
            else:
                bar = "-" * bar_len
            print(f"    {genre:<8s}  {d:+.4f}  {bar}")
        print(f"  {'='*71}")
        print()

        # -- Assertions --

        action_delta = final_avgs["Action"] - initial_avgs["Action"]
        romance_delta = final_avgs["Romance"] - initial_avgs["Romance"]

        # After liking action movies, action probe avg should rise more than romance
        assert action_delta > romance_delta, (
            f"Action probes should rise more than romance: "
            f"action={action_delta:+.4f}, romance={romance_delta:+.4f}"
        )

        # Action scores should have increased (4 action/sci-fi likes)
        assert action_delta > 0, (
            f"Action probes should rise after liking action movies: {action_delta:+.4f}"
        )

        # Romance scores should have decreased (1 romance dislike, no romance likes)
        assert romance_delta < 0, (
            f"Romance probes should drop after disliking a romance: {romance_delta:+.4f}"
        )

        # Skip should produce exactly zero score change
        assert scores_before_skip is not None, "Skip step was not reached"
        for key in scores_after_skip:
            assert scores_after_skip[key] == scores_before_skip[key], (
                f"Skip should not change scores, but {key} changed: "
                f"{scores_before_skip[key]:+.4f} -> {scores_after_skip[key]:+.4f}"
            )

        # Vector should have drifted from the cold-start position
        assert drift > 0

        # All swiped movies should be in the seen set
        assert len(seen) == len(swipe_plan)

        # Online vector should be stored
        assert USER in rec.online_user_vectors
