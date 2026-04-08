from collections.abc import MutableMapping
from logging import Logger

import numpy as np

from movie_recommender.schemas.requests.interactions import SwipeAction
from movie_recommender.services.recommender.serving.artifact_loader import (
    RecommenderArtifacts,
)
from movie_recommender.services.recommender.serving.feedback_mapping import (
    swipe_to_preference,
)
from movie_recommender.services.recommender.serving.online_updater import (
    update_user_vector,
)
from movie_recommender.services.recommender.serving.user_vectors import (
    current_user_vector,
)


def apply_feedback_update(
    artifacts: RecommenderArtifacts,
    online_user_vectors: MutableMapping[str, np.ndarray],
    user_seen_movie_ids: MutableMapping[str, set[int]],
    user_id: str,
    movie_id: int,
    interaction_type: SwipeAction,
    is_supercharged: bool,
    eta: float,
    norm_cap: float,
    logger: Logger,
) -> None:
    preference = swipe_to_preference(interaction_type, is_supercharged)
    user_seen_movie_ids.setdefault(user_id, set()).add(int(movie_id))

    movie_index = artifacts.movie_id_to_index.get(int(movie_id))
    if movie_index is None:
        logger.warning(
            "Movie id %s not found in embeddings; skipping vector update",
            movie_id,
        )
        return

    user_vector = current_user_vector(
        artifacts=artifacts,
        online_user_vectors=online_user_vectors,
        user_id=user_id,
    )
    movie_vector = artifacts.movie_embeddings[movie_index]
    online_user_vectors[user_id] = update_user_vector(
        user_vector=user_vector,
        movie_vector=movie_vector,
        preference=preference,
        eta=eta,
        norm_cap=norm_cap,
    )
