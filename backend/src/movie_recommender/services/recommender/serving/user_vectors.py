from typing import Dict, List, Optional

import numpy as np

from movie_recommender.services.recommender.serving.artifact_loader import (
    RecommenderArtifacts,
)


def to_int_user_id(user_id: str) -> Optional[int]:
    try:
        return int(user_id)
    except (TypeError, ValueError):
        return None


def cold_start_vector(artifacts: RecommenderArtifacts) -> np.ndarray:
    return artifacts.user_embeddings.mean(axis=0).astype(np.float32, copy=False)


def base_user_vector(artifacts: RecommenderArtifacts, user_id: str) -> np.ndarray:
    parsed_user_id = to_int_user_id(user_id)
    if parsed_user_id is not None and parsed_user_id in artifacts.user_id_to_index:
        user_index = artifacts.user_id_to_index[parsed_user_id]
        return artifacts.user_embeddings[user_index].astype(np.float32, copy=False)
    return cold_start_vector(artifacts)


def warm_start_vector(
    artifacts: RecommenderArtifacts, movie_ids: List[int]
) -> np.ndarray:
    """Compute a user vector from a set of liked movie IDs (onboarding picks)."""
    indices = [
        artifacts.movie_id_to_index[mid]
        for mid in movie_ids
        if mid in artifacts.movie_id_to_index
    ]
    if not indices:
        return cold_start_vector(artifacts)
    return (
        artifacts.movie_embeddings[indices].mean(axis=0).astype(np.float32, copy=False)
    )


def current_user_vector(
    artifacts: RecommenderArtifacts,
    online_user_vectors: Dict[str, np.ndarray],
    user_id: str,
) -> np.ndarray:
    if user_id in online_user_vectors:
        return online_user_vectors[user_id]
    return base_user_vector(artifacts, user_id)
