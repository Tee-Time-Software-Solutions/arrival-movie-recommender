import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from movie_recommender.services.recommender.main import Recommender
from movie_recommender.services.recommender.pipeline.online.artifacts import (
    RecommenderArtifacts,
)
from movie_recommender.services.recommender.pipeline.online.serving.user_state import (
    cold_start_vector,
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
        movie_id_to_genres={
            100: ["Action"],
            101: ["Comedy"],
            102: ["Drama"],
            103: ["Action", "Comedy"],
            104: ["Horror"],
        },
        all_movie_ids=np.array([100, 101, 102, 103, 104], dtype=np.int32),
    )


@pytest.fixture
def recommender(synthetic_artifacts: RecommenderArtifacts) -> Recommender:
    """
    Recommender pre-loaded with synthetic artifacts.
    - _get_user_vector always returns cold start (no Redis/DB I/O)
    - _redis is an AsyncMock with smembers returning empty set by default
    - _persist_vector_to_db is a no-op AsyncMock
    """
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.smembers = AsyncMock(return_value=set())
    mock_redis.hgetall = AsyncMock(return_value={})
    mock_redis.incr = AsyncMock(return_value=1)
    mock_redis.expire = AsyncMock()
    mock_redis.set = AsyncMock()
    mock_redis.sadd = AsyncMock()

    rec = Recommender.__new__(Recommender)
    rec.model_artifacts = synthetic_artifacts
    rec.learning_rate = 0.05
    rec.adaptive_learning_strength = 0.0
    rec.norm_cap = 10.0
    rec.exploration_weight = 0.0
    rec.diversity_weight = 0.0
    rec.graph_weight = 0.0
    rec.graph_rerank_top_k = 200
    rec._redis = mock_redis
    rec._neo4j_driver = None
    rec._db_session_factory = MagicMock()

    async def _get_user_vector(user_id: int) -> np.ndarray:
        return cold_start_vector(synthetic_artifacts)

    rec._get_user_vector = _get_user_vector
    rec._persist_vector_to_db = AsyncMock()

    return rec
