import logging

import numpy as np

from movie_recommender.schemas.requests.interactions import SwipeAction
from movie_recommender.services.recommender.pipeline.online.artifacts import (
    RecommenderArtifacts,
)
from movie_recommender.services.recommender.pipeline.online.learning.updater import (
    update_user_vector,
)

logger = logging.getLogger(__name__)


def swipe_to_preference(action_type: SwipeAction, is_supercharged: bool) -> int:
    if action_type == SwipeAction.SKIP:
        return 0
    elif action_type == SwipeAction.DISLIKE:
        return -2 if is_supercharged else -1
    elif action_type == SwipeAction.LIKE:
        return 2 if is_supercharged else 1
    raise ValueError(f"Unsupported swipe action: {action_type}")


def apply_feedback_update(
    model_artifacts: RecommenderArtifacts,
    user_vector: np.ndarray,
    movie_id: int,
    interaction_type: SwipeAction,
    is_supercharged: bool,
    learning_rate: float,
    norm_cap: float,
) -> np.ndarray | None:
    """Returns updated user vector, or None if movie is not in the embeddings."""
    preference = swipe_to_preference(interaction_type, is_supercharged)

    movie_index = model_artifacts.movie_id_to_index.get(int(movie_id))
    if movie_index is None:
        logger.warning(
            "Movie id %s not found in embeddings; skipping vector update", movie_id
        )
        return None

    movie_vector = model_artifacts.movie_embeddings[movie_index]
    return update_user_vector(
        user_vector=user_vector,
        movie_vector=movie_vector,
        preference=preference,
        learning_rate=learning_rate,
        norm_cap=norm_cap,
    )
