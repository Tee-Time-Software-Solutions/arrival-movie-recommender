import numpy as np

from movie_recommender.services.recommender.pipeline.online.artifacts import (
    RecommenderArtifacts,
)


def cold_start_vector(model_artifacts: RecommenderArtifacts) -> np.ndarray:
    """Mean of all trained user embeddings — fallback for users not in the training set."""
    return model_artifacts.user_embeddings.mean(axis=0).astype(np.float32, copy=False)


def base_user_vector(model_artifacts: RecommenderArtifacts, user_id: int) -> np.ndarray:
    """Return the ALS-trained embedding for a known user, or cold start for new ones."""
    idx = model_artifacts.user_id_to_index.get(user_id)
    if idx is None:
        return cold_start_vector(model_artifacts)
    return model_artifacts.user_embeddings[idx].copy()
