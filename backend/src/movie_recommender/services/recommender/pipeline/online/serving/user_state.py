import numpy as np

from movie_recommender.services.recommender.pipeline.online.artifacts import (
    RecommenderArtifacts,
)


def cold_start_vector(model_artifacts: RecommenderArtifacts) -> np.ndarray:
    return model_artifacts.user_embeddings.mean(axis=0).astype(np.float32, copy=False)
