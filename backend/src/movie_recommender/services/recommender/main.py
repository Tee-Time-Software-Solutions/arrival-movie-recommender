from typing import List, Optional, Tuple

from movie_recommender.schemas.users import UserPreferences
from movie_recommender.services.recommender.serving.artifact_loader import (
    RecommenderArtifacts,
    load_recommender_artifacts,
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

        try:
            self.artifacts = load_recommender_artifacts()
        except FileNotFoundError as exc:
            self._artifact_load_error = str(exc)

    def get_top_n(
        self, user_id: str, n: int, user_preferences: Optional[UserPreferences]
    ) -> List[Tuple[int, str]]:
        """
        Def: given a user_id and number of movies to retrieve it returns a list of IDs of movies. These movie IDs must
             may be provided by the recommeder. They must be unique and not clash with existing ones in the db.
            The n returned  movies must respect user preferneces defined in the paremeter 'user_preferences'
        """
        del user_preferences

        if self.artifacts is None:
            raise RuntimeError(
                "Recommender artifacts are not available. "
                f"Details: {self._artifact_load_error}"
            )

        parsed_user_id = _to_int_user_id(user_id)

        if (
            parsed_user_id is not None
            and parsed_user_id in self.artifacts.user_id_to_index
        ):
            user_index = self.artifacts.user_id_to_index[parsed_user_id]
            user_vector = self.artifacts.user_embeddings[user_index]
        else:
            # Cold-start baseline for this commit: mean user vector.
            user_vector = self.artifacts.user_embeddings.mean(axis=0)

        scores = self.artifacts.movie_embeddings @ user_vector
        top_indices = _top_n_indices(scores, n)

        recommendations: List[Tuple[int, str]] = []
        for movie_index in top_indices:
            movie_id = self.artifacts.index_to_movie_id[int(movie_index)]
            movie_title = self.artifacts.movie_id_to_title.get(
                movie_id, f"movie_{movie_id}"
            )
            recommendations.append((movie_id, movie_title))

        return recommendations

    # missing update_user() and similar_movies()
