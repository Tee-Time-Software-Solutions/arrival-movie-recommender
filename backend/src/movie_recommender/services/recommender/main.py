from typing import Dict, List, Optional, Tuple

import numpy as np

from movie_recommender.schemas.interactions import SwipeAction
from movie_recommender.schemas.users import UserPreferences
from movie_recommender.services.recommender.serving.artifact_loader import (
    RecommenderArtifacts,
    load_recommender_artifacts,
)
from movie_recommender.services.recommender.serving.feedback_mapping import (
    swipe_to_preference,
)
from movie_recommender.services.recommender.serving.online_updater import (
    update_user_vector,
)


def _to_int_user_id(user_id: str) -> Optional[int]:
    try:
        return int(user_id)
    except (TypeError, ValueError):
        return None


def _top_n_indices(scores, n: int):
    if n <= 0 or len(scores) == 0:
        return []

    n = min(n, len(scores))
    indices = scores.argpartition(-n)[-n:]
    return indices[scores[indices].argsort()[::-1]].tolist()


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

    def _require_artifacts(self) -> RecommenderArtifacts:
        if self.artifacts is None:
            raise RuntimeError(
                "Recommender artifacts are not available. "
                f"Details: {self._artifact_load_error}"
            )
        return self.artifacts

    def _cold_start_vector(self, artifacts: RecommenderArtifacts) -> np.ndarray:
        return artifacts.user_embeddings.mean(axis=0).astype(np.float32, copy=False)

    def _base_user_vector(self, user_id: str) -> np.ndarray:
        artifacts = self._require_artifacts()
        parsed_user_id = _to_int_user_id(user_id)
        if (
            parsed_user_id is not None
            and parsed_user_id in artifacts.user_id_to_index
        ):
            user_index = artifacts.user_id_to_index[parsed_user_id]
            return artifacts.user_embeddings[user_index].astype(np.float32, copy=False)
        return self._cold_start_vector(artifacts)

    def _current_user_vector(self, user_id: str) -> np.ndarray:
        if user_id in self.online_user_vectors:
            return self.online_user_vectors[user_id]
        return self._base_user_vector(user_id)

    def get_top_n(
        self, user_id: str, n: int, user_preferences: Optional[UserPreferences]
    ) -> List[Tuple[int, str]]:
        """
        Def: given a user_id and number of movies to retrieve it returns a list of IDs of movies. These movie IDs must
             may be provided by the recommeder. They must be unique and not clash with existing ones in the db.
            The n returned  movies must respect user preferneces defined in the paremeter 'user_preferences'
        """
        del user_preferences

        artifacts = self._require_artifacts()
        user_vector = self._current_user_vector(user_id)

        scores = artifacts.movie_embeddings @ user_vector
        seen_movies = self.user_seen_movie_ids.get(user_id, set())
        seen_indices = [
            artifacts.movie_id_to_index[movie_id]
            for movie_id in seen_movies
            if movie_id in artifacts.movie_id_to_index
        ]

        if seen_indices:
            scores = scores.copy()
            scores[seen_indices] = -np.inf

        candidate_indices = np.where(np.isfinite(scores))[0]
        if len(candidate_indices) == 0:
            return []

        candidate_scores = scores[candidate_indices]
        top_local_indices = _top_n_indices(candidate_scores, n)
        top_indices = candidate_indices[top_local_indices].tolist()

        recommendations: List[Tuple[int, str]] = []
        for movie_index in top_indices:
            movie_id = artifacts.index_to_movie_id[int(movie_index)]
            movie_title = artifacts.movie_id_to_title.get(
                movie_id, f"movie_{movie_id}"
            )
            recommendations.append((movie_id, movie_title))

        return recommendations

    def update_user(
        self,
        user_id: str,
        movie_id: int,
        action_type: SwipeAction,
        is_supercharged: bool = False,
    ) -> None:
        artifacts = self._require_artifacts()
        preference = swipe_to_preference(action_type, is_supercharged)
        self.user_seen_movie_ids.setdefault(user_id, set()).add(int(movie_id))

        movie_index = artifacts.movie_id_to_index.get(int(movie_id))
        if movie_index is None:
            return

        current_user_vector = self._current_user_vector(user_id)
        movie_vector = artifacts.movie_embeddings[movie_index]
        updated_user_vector = update_user_vector(
            user_vector=current_user_vector,
            movie_vector=movie_vector,
            preference=preference,
            eta=self.eta,
            norm_cap=self.norm_cap,
        )
        self.online_user_vectors[user_id] = updated_user_vector

    # missing update_user() and similar_movies()
