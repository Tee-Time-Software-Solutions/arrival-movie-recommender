import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from movie_recommender.schemas.interactions import SwipeAction

logger = logging.getLogger(__name__)
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
        self, user_id: str, n: int, user_preferences: UserPreferences
    ) -> List[Tuple[int, str]]:
        """
        Def: given a user_id and number of movies to retrieve it returns a list of IDs of movies. These movie IDs must
             may be provided by the recommeder. They must be unique and not clash with existing ones in the db.
            The n returned  movies must respect user preferneces defined in the paremeter 'user_preferences'

        TODO:
            change user_preferences param to 'list_of_filtered_movies: List[ids:int]'
        """
        del user_preferences

        artifacts = self._require_artifacts()
        user_vector = self._current_user_vector(user_id)
        vector_norm = float(np.linalg.norm(user_vector))
        is_online = user_id in self.online_user_vectors

        logger.info(
            f"\n{'='*60}\n"
            f"  RECOMMENDER | get_top_n(user={user_id}, n={n})\n"
            f"  Vector source: {'ONLINE (updated by swipes)' if is_online else 'COLD START (mean of all users)'}\n"
            f"  Vector L2 norm: {vector_norm:.4f}\n"
            f"  Movies seen: {len(self.user_seen_movie_ids.get(user_id, set()))}\n"
            f"{'='*60}"
        )

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

        logger.info(
            f"  TOP {n} RECOMMENDATIONS:\n"
            + "\n".join(
                f"    {i+1}. [{mid}] {title} (score: {float(scores[artifacts.movie_id_to_index[mid]]):.4f})"
                for i, (mid, title) in enumerate(recommendations)
            )
        )

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

        movie_title = artifacts.movie_id_to_title.get(int(movie_id), f"movie_{movie_id}")

        movie_index = artifacts.movie_id_to_index.get(int(movie_id))
        if movie_index is None:
            logger.warning(f"  SWIPE | movie_id={movie_id} not in embeddings, skipping update")
            return

        current_user_vector = self._current_user_vector(user_id)
        old_norm = float(np.linalg.norm(current_user_vector))

        movie_vector = artifacts.movie_embeddings[movie_index]
        updated_user_vector = update_user_vector(
            user_vector=current_user_vector,
            movie_vector=movie_vector,
            preference=preference,
            eta=self.eta,
            norm_cap=self.norm_cap,
        )
        self.online_user_vectors[user_id] = updated_user_vector

        new_norm = float(np.linalg.norm(updated_user_vector))
        vector_diff = updated_user_vector - current_user_vector
        vector_delta = float(np.linalg.norm(vector_diff))

        # Show which dimensions changed most
        abs_diff = np.abs(vector_diff)
        top_dims = np.argsort(abs_diff)[-5:][::-1]
        dim_changes = "\n".join(
            f"    dim[{d:>2}]: {current_user_vector[d]:+.4f} -> {updated_user_vector[d]:+.4f} ({vector_diff[d]:+.6f})"
            for d in top_dims
        )

        # Cosine similarity: how aligned is user now with this movie?
        cos_before = float(np.dot(current_user_vector, movie_vector) / (old_norm * np.linalg.norm(movie_vector) + 1e-9))
        cos_after = float(np.dot(updated_user_vector, movie_vector) / (new_norm * np.linalg.norm(movie_vector) + 1e-9))

        # Score change for this movie
        score_before = float(np.dot(current_user_vector, movie_vector))
        score_after = float(np.dot(updated_user_vector, movie_vector))

        # Show impact: top 3 movies that gained/lost the most score
        all_scores_before = artifacts.movie_embeddings @ current_user_vector
        all_scores_after = artifacts.movie_embeddings @ updated_user_vector
        score_changes = all_scores_after - all_scores_before

        biggest_gainers_idx = np.argsort(score_changes)[-3:][::-1]
        biggest_losers_idx = np.argsort(score_changes)[:3]

        gainers = "\n".join(
            f"    + {artifacts.movie_id_to_title.get(artifacts.index_to_movie_id[int(i)], '?')[:35]:35s} ({score_changes[i]:+.6f})"
            for i in biggest_gainers_idx
        )
        losers = "\n".join(
            f"    - {artifacts.movie_id_to_title.get(artifacts.index_to_movie_id[int(i)], '?')[:35]:35s} ({score_changes[i]:+.6f})"
            for i in biggest_losers_idx
        )

        action_str = 'LIKED' if action_type == SwipeAction.LIKE else 'DISLIKED' if action_type == SwipeAction.DISLIKE else 'SKIPPED'

        logger.info(
            f"\n{'~'*60}\n"
            f"  SWIPE | user={user_id} {action_str}"
            f"{' (SUPERCHARGED)' if is_supercharged else ''}"
            f" \"{movie_title}\" (id={movie_id})\n"
            f"  Preference weight: {preference:+.2f} | eta: {self.eta}\n"
            f"\n"
            f"  VECTOR UPDATE:\n"
            f"    Norm: {old_norm:.4f} -> {new_norm:.4f}\n"
            f"    Delta (L2): {vector_delta:.6f}\n"
            f"    Cosine sim with \"{movie_title[:25]}\": {cos_before:.4f} -> {cos_after:.4f}\n"
            f"    Score for this movie: {score_before:.4f} -> {score_after:.4f}\n"
            f"\n"
            f"  TOP 5 DIMENSIONS CHANGED:\n"
            f"{dim_changes}\n"
            f"\n"
            f"  BIGGEST GAINERS (movies now ranked higher):\n"
            f"{gainers}\n"
            f"  BIGGEST LOSERS (movies now ranked lower):\n"
            f"{losers}\n"
            f"\n"
            f"  Total swipes: {len(self.user_seen_movie_ids.get(user_id, set()))}\n"
        )

        # Show updated top N recommendations
        seen_movies = self.user_seen_movie_ids.get(user_id, set())
        top_n_scores = all_scores_after.copy()
        seen_indices = [
            artifacts.movie_id_to_index[mid]
            for mid in seen_movies
            if mid in artifacts.movie_id_to_index
        ]
        if seen_indices:
            top_n_scores[seen_indices] = -np.inf

        top_n = 10
        candidate_indices = np.where(np.isfinite(top_n_scores))[0]
        candidate_scores = top_n_scores[candidate_indices]
        top_local = _top_n_indices(candidate_scores, top_n)
        top_indices = candidate_indices[top_local].tolist()

        recs = "\n".join(
            f"    {i+1:>2}. [{artifacts.index_to_movie_id[int(idx)]}] "
            f"{artifacts.movie_id_to_title.get(artifacts.index_to_movie_id[int(idx)], '?')[:40]:40s} "
            f"(score: {all_scores_after[idx]:.4f})"
            for i, idx in enumerate(top_indices)
        )

        logger.info(
            f"  NEXT TOP {top_n} RECOMMENDATIONS (after update):\n"
            f"{recs}\n"
            f"{'~'*60}"
        )

    def set_user_feedback(
        self,
        user_id: str,
        movie_id: int,
        interaction_type: SwipeAction,
        is_supercharged: bool = False,
    ) -> None:
        """Adapter for the endpoint layer — delegates to update_user()."""
        self.update_user(
            user_id=user_id,
            movie_id=movie_id,
            action_type=interaction_type,
            is_supercharged=is_supercharged,
        )

    def get_similar_n_movies(self, movie_id: int, n: int):
        """
        Def: Give a movie_id returns its closes movies

        TODO:
            - Implement after MVP
        """
