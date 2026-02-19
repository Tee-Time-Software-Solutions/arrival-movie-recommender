"""
Full offline-to-online pipeline integration test.

Generates synthetic MovieLens-format CSV data, runs the entire offline pipeline
(preprocess -> filter -> split -> build matrix -> train ALS -> evaluate), then
verifies the produced artifacts can be loaded by the online recommender and
used for recommendations + swipe feedback.

All file I/O is redirected to a temporary directory via monkey-patching.
The actual pipeline code is NOT modified.

Run with:  pytest tests/integration/test_full_pipeline.py -v -s
"""

import io
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from movie_recommender.schemas.interactions import SwipeAction
from movie_recommender.services.recommender.main import Recommender
from movie_recommender.services.recommender.serving.artifact_loader import (
    RecommenderArtifacts,
    load_recommender_artifacts,
)

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

NUM_USERS = 20
NUM_MOVIES = 15


def _generate_movies_csv(path: Path) -> None:
    """Create a minimal movies.csv in MovieLens format."""
    genres_pool = [
        "Action|Adventure",
        "Comedy|Romance",
        "Drama",
        "Horror|Thriller",
        "Sci-Fi|Action",
        "Comedy",
        "Drama|Romance",
        "Action",
        "Thriller",
        "Animation|Comedy",
        "Documentary",
        "Crime|Drama",
        "Fantasy|Adventure",
        "Horror",
        "Musical|Comedy",
    ]
    rows = []
    for i in range(NUM_MOVIES):
        movie_id = 1000 + i
        title = f"Test Movie {i} ({2000 + i})"
        genres = genres_pool[i % len(genres_pool)]
        rows.append(f"{movie_id},{title},{genres}")

    header = "movieId,title,genres"
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


def _generate_ratings_csv(path: Path) -> None:
    """Create ratings.csv where every user rates every movie.

    20 users x 15 movies = 300 interactions.
    Each user has 15 ratings  (>= 10 threshold).
    Each movie has 20 ratings (>= 20 threshold).
    Ratings span 0.5-5.0 to cover all preference buckets.
    Timestamps are sequential so chronological split works.
    """
    rng = np.random.RandomState(42)
    possible_ratings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    rows = []
    ts = 1_000_000_000  # base timestamp

    for user_id in range(1, NUM_USERS + 1):
        for movie_id in range(1000, 1000 + NUM_MOVIES):
            rating = rng.choice(possible_ratings)
            rows.append(f"{user_id},{movie_id},{rating},{ts}")
            ts += 100  # increment so every interaction has a unique timestamp

    header = "userId,movieId,rating,timestamp"
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


# ---------------------------------------------------------------------------
# Path-patching helpers
# ---------------------------------------------------------------------------

# Every pipeline module imports paths at module load time and stores them as
# module-level constants.  We need to patch each one individually.

_MODULE_PREFIX = "movie_recommender.services.recommender"

_PATH_PATCHES = {
    # paths_dev.py base paths
    f"{_MODULE_PREFIX}.paths_dev.DATA_RAW": "data/raw",
    f"{_MODULE_PREFIX}.paths_dev.DATA_PROCESSED": "data/processed",
    f"{_MODULE_PREFIX}.paths_dev.DATA_SPLITS": "data/splits",
    f"{_MODULE_PREFIX}.paths_dev.ARTIFACTS": "artifacts",
    # preprocess_movies.py
    f"{_MODULE_PREFIX}.data_processing.preprocessing.preprocess_movies.RAW_PATH": "data/raw/movies.csv",
    f"{_MODULE_PREFIX}.data_processing.preprocessing.preprocess_movies.PROCESSED_PATH": "data/processed/movies_clean.parquet",
    # preprocess_ratings.py
    f"{_MODULE_PREFIX}.data_processing.preprocessing.preprocess_ratings.RAW_PATH": "data/raw/ratings.csv",
    f"{_MODULE_PREFIX}.data_processing.preprocessing.preprocess_ratings.PROCESSED_PATH": "data/processed/interactions_clean.parquet",
    # filtering.py
    f"{_MODULE_PREFIX}.data_processing.preprocessing.filtering.PROCESSED_INPUT": "data/processed/interactions_clean.parquet",
    f"{_MODULE_PREFIX}.data_processing.preprocessing.filtering.PROCESSED_OUTPUT": "data/processed/interactions_filtered.parquet",
    # prune_movies.py
    f"{_MODULE_PREFIX}.data_processing.preprocessing.prune_movies.MOVIES_INPUT": "data/processed/movies_clean.parquet",
    f"{_MODULE_PREFIX}.data_processing.preprocessing.prune_movies.MOVIES_OUTPUT": "data/processed/movies_filtered.parquet",
    f"{_MODULE_PREFIX}.data_processing.preprocessing.prune_movies.INTERACTIONS_INPUT": "data/processed/interactions_filtered.parquet",
    # split.py
    f"{_MODULE_PREFIX}.data_processing.split.INPUT_PATH": "data/processed/interactions_filtered.parquet",
    f"{_MODULE_PREFIX}.data_processing.split.TRAIN_PATH": "data/splits/train.parquet",
    f"{_MODULE_PREFIX}.data_processing.split.VAL_PATH": "data/splits/val.parquet",
    f"{_MODULE_PREFIX}.data_processing.split.TEST_PATH": "data/splits/test.parquet",
    # build_matrix.py
    f"{_MODULE_PREFIX}.learning.build_matrix.TRAIN_PATH": "data/splits/train.parquet",
    f"{_MODULE_PREFIX}.learning.build_matrix.MATRIX_PATH": "artifacts/R_train.npz",
    f"{_MODULE_PREFIX}.learning.build_matrix.MAPPINGS_PATH": "artifacts/mappings.json",
    # train_als.py
    f"{_MODULE_PREFIX}.learning.train_als.MATRIX_PATH": "artifacts/R_train.npz",
    f"{_MODULE_PREFIX}.learning.train_als.MOVIE_EMBEDDINGS_PATH": "artifacts/movie_embeddings.npy",
    f"{_MODULE_PREFIX}.learning.train_als.USER_EMBEDDINGS_PATH": "artifacts/user_embeddings.npy",
    f"{_MODULE_PREFIX}.learning.train_als.MODEL_INFO_PATH": "artifacts/model_info.json",
    # evaluate.py
    f"{_MODULE_PREFIX}.learning.evaluate.TRAIN_PATH": "data/splits/train.parquet",
    f"{_MODULE_PREFIX}.learning.evaluate.VAL_PATH": "data/splits/val.parquet",
    f"{_MODULE_PREFIX}.learning.evaluate.USER_EMB_PATH": "artifacts/user_embeddings.npy",
    f"{_MODULE_PREFIX}.learning.evaluate.MOVIE_EMB_PATH": "artifacts/movie_embeddings.npy",
    f"{_MODULE_PREFIX}.learning.evaluate.MAPPINGS_PATH": "artifacts/mappings.json",
    # artifact_loader.py (online serving)
    f"{_MODULE_PREFIX}.serving.artifact_loader.MOVIE_EMBEDDINGS_PATH": "artifacts/movie_embeddings.npy",
    f"{_MODULE_PREFIX}.serving.artifact_loader.USER_EMBEDDINGS_PATH": "artifacts/user_embeddings.npy",
    f"{_MODULE_PREFIX}.serving.artifact_loader.MAPPINGS_PATH": "artifacts/mappings.json",
    f"{_MODULE_PREFIX}.serving.artifact_loader.MOVIES_FILTERED_PATH": "data/processed/movies_filtered.parquet",
}


def _build_patches(tmp_dir: Path) -> dict[str, Path]:
    """Resolve all relative paths against the temp directory."""
    return {target: tmp_dir / rel for target, rel in _PATH_PATCHES.items()}


# ---------------------------------------------------------------------------
# Session-scoped fixture: run the pipeline once, reuse for all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pipeline_dir():
    """Create synthetic data, run the full offline pipeline in a temp dir.

    Yields the temp directory Path.  Cleaned up after all tests complete.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="pipeline_test_"))

    try:
        # Create directory structure
        (tmp_dir / "data" / "raw").mkdir(parents=True)
        (tmp_dir / "data" / "processed").mkdir(parents=True)
        (tmp_dir / "data" / "splits").mkdir(parents=True)
        (tmp_dir / "artifacts").mkdir(parents=True)

        # Generate synthetic CSVs
        _generate_movies_csv(tmp_dir / "data" / "raw" / "movies.csv")
        _generate_ratings_csv(tmp_dir / "data" / "raw" / "ratings.csv")

        # Patch every module-level path constant to point at tmp_dir
        patches = _build_patches(tmp_dir)
        context_managers = [
            patch(target, new_value) for target, new_value in patches.items()
        ]

        # Activate all patches
        for cm in context_managers:
            cm.start()

        try:
            # Suppress the pipeline's verbose print output
            from movie_recommender.services.recommender.learning.offline_pipeline import (
                run_pipeline,
            )

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                run_pipeline()
            finally:
                sys.stdout = old_stdout
        finally:
            # Deactivate all patches
            for cm in context_managers:
                cm.stop()

        yield tmp_dir

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def loaded_artifacts(pipeline_dir: Path):
    """Load artifacts from the pipeline output using the real loader."""
    patches = _build_patches(pipeline_dir)
    context_managers = [
        patch(target, new_value) for target, new_value in patches.items()
    ]
    for cm in context_managers:
        cm.start()
    try:
        artifacts = load_recommender_artifacts()
    finally:
        for cm in context_managers:
            cm.stop()
    return artifacts


@pytest.fixture
def pipeline_recommender(loaded_artifacts: RecommenderArtifacts) -> Recommender:
    """Fresh Recommender instance backed by real ALS-trained artifacts."""
    rec = Recommender.__new__(Recommender)
    rec.artifacts = loaded_artifacts
    rec._artifact_load_error = None
    rec.online_user_vectors = {}
    rec.user_seen_movie_ids = {}
    rec.eta = 0.05
    rec.norm_cap = 10.0
    return rec


# ---------------------------------------------------------------------------
# 1. Offline pipeline execution
# ---------------------------------------------------------------------------


class TestOfflinePipelineRuns:
    def test_pipeline_completes(self, pipeline_dir: Path):
        """The full 8-step pipeline ran without raising."""
        print("\n--- OFFLINE PIPELINE ---")
        print(f"  Temp directory: {pipeline_dir}")
        print(f"  Synthetic data: {NUM_USERS} users x {NUM_MOVIES} movies = {NUM_USERS * NUM_MOVIES} interactions")
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
        print(f"  Interactions after filtering: {len(df)}")
        print(f"  (started with {NUM_USERS * NUM_MOVIES})")
        assert len(df) > 0, "All interactions were filtered out!"

    def test_all_users_survived(self, pipeline_dir: Path):
        """With 15 ratings per user (threshold=10), all 20 users should survive."""
        df = pd.read_parquet(
            pipeline_dir / "data/processed/interactions_filtered.parquet"
        )
        n_users = df["user_id"].nunique()
        print(f"\n--- USER SURVIVAL ---")
        print(f"  Users after filtering: {n_users} / {NUM_USERS}")
        print(f"  Ratings per user: {NUM_MOVIES} (threshold: 10)")
        print(f"  All survived: {n_users == NUM_USERS}")
        assert n_users == NUM_USERS

    def test_all_movies_survived(self, pipeline_dir: Path):
        """With 20 ratings per movie (threshold=20), all 15 movies should survive."""
        df = pd.read_parquet(
            pipeline_dir / "data/processed/interactions_filtered.parquet"
        )
        n_movies = df["movie_id"].nunique()
        print(f"\n--- MOVIE SURVIVAL ---")
        print(f"  Movies after filtering: {n_movies} / {NUM_MOVIES}")
        print(f"  Ratings per movie: {NUM_USERS} (threshold: 20)")
        print(f"  All survived: {n_movies == NUM_MOVIES}")
        assert n_movies == NUM_MOVIES

    def test_split_has_train_val_test(self, pipeline_dir: Path):
        train = pd.read_parquet(pipeline_dir / "data/splits/train.parquet")
        val = pd.read_parquet(pipeline_dir / "data/splits/val.parquet")
        test = pd.read_parquet(pipeline_dir / "data/splits/test.parquet")
        filtered = pd.read_parquet(
            pipeline_dir / "data/processed/interactions_filtered.parquet"
        )
        total = len(train) + len(val) + len(test)

        print(f"\n--- CHRONOLOGICAL SPLIT ---")
        print(f"  Train: {len(train)} ({len(train)/total*100:.0f}%)")
        print(f"  Val:   {len(val)} ({len(val)/total*100:.0f}%)")
        print(f"  Test:  {len(test)} ({len(test)/total*100:.0f}%)")
        print(f"  Total: {total} (filtered: {len(filtered)})")
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
            # Show a few sample titles
            sample = list(loaded_artifacts.movie_id_to_title.items())[:3]
            for mid, title in sample:
                print(f"    id={mid}: \"{title}\"")
        assert len(missing) == 0


# ---------------------------------------------------------------------------
# 5. Online recommender with real artifacts
# ---------------------------------------------------------------------------


class TestOnlineWithRealArtifacts:
    def test_known_user_gets_recommendations(self, pipeline_recommender: Recommender):
        """A user that exists in training data gets results."""
        recs = pipeline_recommender.get_top_n(user_id="1", n=5, user_preferences=None)
        print(f"\n--- KNOWN USER RECOMMENDATIONS ---")
        print(f"  User: 1 (in training data)")
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
            user_id="99999", n=5, user_preferences=None
        )
        print(f"\n--- COLD-START USER RECOMMENDATIONS ---")
        print(f"  User: 99999 (NOT in training data)")
        print(f"  Fallback: mean of all user embeddings")
        print(f"  Returned: {len(recs)} recommendations")
        for i, (mid, title) in enumerate(recs, 1):
            print(f"    {i}. {title} (id={mid})")
        assert len(recs) > 0

    def test_recommendation_ids_exist_in_mappings(
        self, pipeline_recommender: Recommender
    ):
        recs = pipeline_recommender.get_top_n(user_id="1", n=5, user_preferences=None)
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
        self, pipeline_recommender: Recommender
    ):
        """With trained ALS embeddings, different users should generally have
        different preferences.  Check that not all users get the same #1 movie."""
        print(f"\n--- PERSONALISATION CHECK ---")
        top_picks = {}
        for uid in range(1, min(NUM_USERS + 1, 11)):
            recs = pipeline_recommender.get_top_n(
                user_id=str(uid), n=1, user_preferences=None
            )
            if recs:
                mid, title = recs[0]
                top_picks[uid] = (mid, title)
                print(f"  User {uid:2d} top pick: {title} (id={mid})")
        distinct = len(set(mid for mid, _ in top_picks.values()))
        print(f"  Distinct top picks across {len(top_picks)} users: {distinct}")
        assert distinct >= 2


# ---------------------------------------------------------------------------
# 6. Full loop: offline -> online -> swipe -> changed recs
# ---------------------------------------------------------------------------


class TestFullLoop:
    def test_swipe_excludes_movie_from_future_recs(
        self, pipeline_recommender: Recommender
    ):
        recs = pipeline_recommender.get_top_n(user_id="1", n=5, user_preferences=None)
        top_movie_id, top_title = recs[0]
        ids_before = [mid for mid, _ in recs]

        pipeline_recommender.update_user(
            user_id="1", movie_id=top_movie_id, action_type=SwipeAction.LIKE
        )

        recs_after = pipeline_recommender.get_top_n(
            user_id="1", n=5, user_preferences=None
        )
        ids_after = [mid for mid, _ in recs_after]

        print(f"\n--- SEEN-MOVIE EXCLUSION ---")
        print(f"  Before swipe: {ids_before}")
        print(f"  Swiped LIKE on: \"{top_title}\" (id={top_movie_id})")
        print(f"  After swipe:  {ids_after}")
        print(f"  Excluded: {top_movie_id not in ids_after}")

        assert top_movie_id not in ids_after

    def test_like_updates_user_vector(self, pipeline_recommender: Recommender):
        vec_before = pipeline_recommender._current_user_vector("1").copy()

        first_rec = pipeline_recommender.get_top_n(
            user_id="1", n=1, user_preferences=None
        )
        movie_id, title = first_rec[0]

        pipeline_recommender.update_user(
            user_id="1", movie_id=movie_id, action_type=SwipeAction.LIKE
        )

        vec_after = pipeline_recommender._current_user_vector("1")
        delta = vec_after - vec_before
        norm_delta = float(np.linalg.norm(delta))

        print(f"\n--- LIKE UPDATES VECTOR ---")
        print(f"  User 1 LIKE on \"{title}\" (id={movie_id})")
        print(f"  Vector norm before: {np.linalg.norm(vec_before):.4f}")
        print(f"  Vector norm after:  {np.linalg.norm(vec_after):.4f}")
        print(f"  Delta norm:         {norm_delta:.6f}")
        print(f"  Vector changed: {norm_delta > 0}")

        assert not np.array_equal(vec_before, vec_after), "Vector should have changed"

    def test_dislike_moves_vector_away(self, pipeline_recommender: Recommender):
        """Disliking a movie should decrease its score for the user."""
        artifacts = pipeline_recommender.artifacts
        recs = pipeline_recommender.get_top_n(user_id="2", n=5, user_preferences=None)
        movie_id, title = recs[-1]  # pick last (lowest-scored)
        movie_idx = artifacts.movie_id_to_index[movie_id]

        vec_before = pipeline_recommender._current_user_vector("2")
        score_before = float(artifacts.movie_embeddings[movie_idx] @ vec_before)

        pipeline_recommender.update_user(
            user_id="2", movie_id=movie_id, action_type=SwipeAction.DISLIKE
        )

        vec_after = pipeline_recommender._current_user_vector("2")
        score_after = float(artifacts.movie_embeddings[movie_idx] @ vec_after)

        print(f"\n--- DISLIKE LOWERS SCORE ---")
        print(f"  User 2 DISLIKE on \"{title}\" (id={movie_id})")
        print(f"  Score before: {score_before:.4f}")
        print(f"  Score after:  {score_after:.4f}")
        print(f"  Decreased:    {score_after < score_before}")

        assert score_after < score_before

    def test_full_session_multiple_swipes(self, pipeline_recommender: Recommender):
        """Simulate a browsing session: get recs, swipe, repeat.
        After several swipes the seen set grows and vector changes."""
        user_id = "5"
        actions = [
            SwipeAction.LIKE,
            SwipeAction.DISLIKE,
            SwipeAction.SKIP,
            SwipeAction.LIKE,
            SwipeAction.DISLIKE,
        ]

        print(f"\n--- FULL BROWSING SESSION (User {user_id}) ---")
        print(f"  Starting vector norm: {np.linalg.norm(pipeline_recommender._current_user_vector(user_id)):.4f}")

        for step in range(5):
            recs = pipeline_recommender.get_top_n(
                user_id=user_id, n=3, user_preferences=None
            )
            if not recs:
                print(f"  Step {step+1}: No more recommendations!")
                break

            movie_id, title = recs[0]
            action = actions[step]
            pipeline_recommender.update_user(
                user_id=user_id, movie_id=movie_id, action_type=action
            )

            vec = pipeline_recommender._current_user_vector(user_id)
            seen = pipeline_recommender.user_seen_movie_ids.get(user_id, set())
            print(f"  Step {step+1}: {action.value.upper():7s} on \"{title}\" (id={movie_id}) | seen={len(seen)} | norm={np.linalg.norm(vec):.4f}")

        seen = pipeline_recommender.user_seen_movie_ids.get(user_id, set())
        remaining = pipeline_recommender.get_top_n(
            user_id=user_id, n=10, user_preferences=None
        )

        print(f"  ---")
        print(f"  Total seen: {len(seen)}")
        print(f"  Remaining recs: {len(remaining)}")
        print(f"  Online vector stored: {user_id in pipeline_recommender.online_user_vectors}")

        assert len(seen) == 5, f"Expected 5 seen movies, got {len(seen)}"
        assert user_id in pipeline_recommender.online_user_vectors
