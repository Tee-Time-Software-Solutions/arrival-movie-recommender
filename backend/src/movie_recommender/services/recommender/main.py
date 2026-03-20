import logging
from typing import Dict, List, Optional

import numpy as np

from movie_recommender.schemas.interactions import SwipeAction
from movie_recommender.services.recommender.serving.artifact_loader import (
    RecommenderArtifacts,
    load_recommender_artifacts,
)
from movie_recommender.services.recommender.serving.feedback_service import (
    apply_feedback_update,
)
from movie_recommender.services.recommender.serving.ranker import rank_movie_ids
from movie_recommender.services.recommender.serving.user_vectors import (
    current_user_vector,
)
from movie_recommender.services.recommender.serving.validation import (
    require_artifacts,
)

logger = logging.getLogger(__name__)


class Recommender:
    def __init__(self) -> None:
        self.artifacts: Optional[RecommenderArtifacts] = None
        self._artifact_load_error: Optional[str] = None
        self.online_user_vectors: Dict[str, np.ndarray] = {}
        self.user_seen_movie_ids: Dict[str, set[int]] = {}
        self.eta = 0.05
        self.norm_cap = 10.0

        try:
            self.artifacts = load_recommender_artifacts()
        except FileNotFoundError as exc:
            self._artifact_load_error = str(exc)

    def get_top_n_recommendations(
        self, user_id: str, list_of_movie_ids: List[int]
    ) -> List[int]:
        """
        Receives a user id and a list of movie ids.
        Returns the movie ids ranked by predicted preference for the user.
        """
        artifacts = require_artifacts(self.artifacts, self._artifact_load_error)
        user_vector = current_user_vector(
            artifacts=artifacts,
            online_user_vectors=self.online_user_vectors,
            user_id=user_id,
        )
        seen_movie_ids = self.user_seen_movie_ids.get(user_id, set())

        return rank_movie_ids(
            artifacts=artifacts,
            user_vector=user_vector,
            movie_ids=list_of_movie_ids,
            seen_movie_ids=seen_movie_ids,
        )

    def set_user_feedback(
        self,
        user_id: str,
        movie_id: int,
        interaction_type: SwipeAction,
        is_supercharged: bool,
    ) -> None:
        """
        Receives a user id, movie id, interaction type, and supercharged flag.
        Updates the user vector based on the feedback. Returns nothing.
        """
        artifacts = require_artifacts(self.artifacts, self._artifact_load_error)
        apply_feedback_update(
            artifacts=artifacts,
            online_user_vectors=self.online_user_vectors,
            user_seen_movie_ids=self.user_seen_movie_ids,
            user_id=user_id,
            movie_id=movie_id,
            interaction_type=interaction_type,
            is_supercharged=is_supercharged,
            eta=self.eta,
            norm_cap=self.norm_cap,
            logger=logger,
        )
