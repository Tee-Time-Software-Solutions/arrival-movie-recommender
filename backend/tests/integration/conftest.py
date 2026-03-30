"""
Shared fixtures for integration tests.

Provides session-scoped fixtures that download the MovieLens 20M dataset
(if needed), run the offline pipeline (if artifacts don't exist), and
load the trained artifacts for use by all integration test modules.
"""

import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import pytest

from movie_recommender.services.recommender.main import Recommender
from movie_recommender.services.recommender.paths_dev import (
    ARTIFACTS,
    DATA_PROCESSED,
    DATA_RAW,
    DATA_SPLITS,
    PROJECT_ROOT,
)
from movie_recommender.services.recommender.serving.artifact_loader import (
    RecommenderArtifacts,
    load_recommender_artifacts,
)

# ---------------------------------------------------------------------------
# MovieLens 20M download helper
# ---------------------------------------------------------------------------

ML20M_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"


def _download_movielens_20m(raw_dir: Path):
    """Download and extract MovieLens 20M if not present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ml-20m.zip"

    print(f"Downloading MovieLens 20M from {ML20M_URL}...")
    import certifi
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(ML20M_URL, context=ssl_ctx) as resp, \
         open(zip_path, "wb") as out:
        shutil.copyfileobj(resp, out)

    with zipfile.ZipFile(zip_path) as zf:
        for name in ["ml-20m/movies.csv", "ml-20m/ratings.csv"]:
            member = zf.getinfo(name)
            target = raw_dir / Path(name).name
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

    zip_path.unlink()  # clean up zip
    print("Download complete.")


# ---------------------------------------------------------------------------
# Session-scoped fixture: download data + run pipeline (cached)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pipeline_dir():
    """Ensure real MovieLens 20M data is present and pipeline has run.

    - Downloads dataset if movies.csv / ratings.csv are missing
    - Runs the offline pipeline if artifacts don't exist yet
    - Yields PROJECT_ROOT; artifacts persist for future runs
    """
    # Step 1: ensure raw CSVs exist
    movies_csv = DATA_RAW / "movies.csv"
    ratings_csv = DATA_RAW / "ratings.csv"

    if not movies_csv.exists() or not ratings_csv.exists():
        _download_movielens_20m(DATA_RAW)

    assert movies_csv.exists(), f"Missing {movies_csv}"
    assert ratings_csv.exists(), f"Missing {ratings_csv}"

    # Step 2: ensure directory structure
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DATA_SPLITS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # Step 3: run pipeline if artifacts missing
    if (ARTIFACTS / "movie_embeddings.npy").exists():
        print("Artifacts already exist, skipping pipeline run.")
    else:
        print("Running offline pipeline (this takes ~5 minutes)...")
        from movie_recommender.services.recommender.learning.offline_pipeline import (
            run_pipeline,
        )

        run_pipeline()
        print("Pipeline complete.")

    yield PROJECT_ROOT


@pytest.fixture(scope="session")
def loaded_artifacts(pipeline_dir: Path):
    """Load artifacts from the pipeline output using the real loader."""
    return load_recommender_artifacts()


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
