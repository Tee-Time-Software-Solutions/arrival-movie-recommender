import numpy as np
import pytest

from movie_recommender.services.recommender.main import Recommender
from movie_recommender.services.recommender.serving.artifact_loader import (
    RecommenderArtifacts,
)


@pytest.fixture
def synthetic_artifacts() -> RecommenderArtifacts:
    """
    Small deterministic artifacts with orthogonal embeddings.

    Movies (5 movies, embedding dim=4):
        index 0 → movie_id 100  "Action Movie"     [1, 0, 0, 0]
        index 1 → movie_id 101  "Comedy Movie"     [0, 1, 0, 0]
        index 2 → movie_id 102  "Drama Movie"      [0, 0, 1, 0]
        index 3 → movie_id 103  "Action Comedy"    [1, 1, 0, 0] (unnormalised)
        index 4 → movie_id 104  "Horror Movie"     [0, 0, 0, 1]

    Users (3 users, embedding dim=4):
        index 0 → user_id 1  "action fan"   [1, 0, 0, 0]
        index 1 → user_id 2  "comedy fan"   [0, 1, 0, 0]
        index 2 → user_id 3  "mixed"        [0.5, 0.5, 0, 0]

    Dot-product scores for user 1 (action fan):
        movie 100: 1.0,  movie 101: 0.0,  movie 102: 0.0,
        movie 103: 1.0,  movie 104: 0.0
    """
    movie_embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    user_embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    return RecommenderArtifacts(
        movie_embeddings=movie_embeddings,
        user_embeddings=user_embeddings,
        user_id_to_index={1: 0, 2: 1, 3: 2},
        movie_id_to_index={100: 0, 101: 1, 102: 2, 103: 3, 104: 4},
        index_to_movie_id={0: 100, 1: 101, 2: 102, 3: 103, 4: 104},
        movie_id_to_title={
            100: "Action Movie",
            101: "Comedy Movie",
            102: "Drama Movie",
            103: "Action Comedy",
            104: "Horror Movie",
        },
    )


@pytest.fixture
def recommender(synthetic_artifacts: RecommenderArtifacts) -> Recommender:
    """Recommender pre-loaded with synthetic artifacts (bypasses disk I/O)."""
    rec = Recommender.__new__(Recommender)
    rec.artifacts = synthetic_artifacts
    rec._artifact_load_error = None
    rec.online_user_vectors = {}
    rec.user_seen_movie_ids = {}
    rec.eta = 0.05
    rec.norm_cap = 10.0
    return rec
