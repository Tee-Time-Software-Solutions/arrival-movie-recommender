import numpy as np

from movie_recommender.services.recommender.pipeline.online.artifacts import (
    RecommenderArtifacts,
)


def rank_movie_ids(
    n: int,
    model_artifacts: RecommenderArtifacts,
    user_vector: np.ndarray,
    seen_movie_ids: set[int],
    genre_impression_counts: dict[str, int] | None = None,
    exploration_weight: float = 0.0,
) -> list[int]:
    """Rank all known movies by score, excluding seen ones."""
    if n == 0:
        return []

    all_ids = model_artifacts.all_movie_ids  # shape (N,), int32

    if seen_movie_ids:
        seen = np.fromiter(seen_movie_ids, dtype=np.int32, count=len(seen_movie_ids))
        mask = ~np.isin(all_ids, seen)
    else:
        mask = np.ones(len(all_ids), dtype=bool)

    candidate_ids = all_ids[mask]
    if len(candidate_ids) == 0:
        return []

    candidate_embeddings = model_artifacts.movie_embeddings[mask]
    scores = candidate_embeddings @ user_vector
    final_scores = scores.copy()

    if exploration_weight > 0:
        candidate_bonus = np.array(
            [
                _genre_exploration_bonus(
                    model_artifacts.movie_id_to_genres.get(int(movie_id), []),
                    genre_impression_counts or {},
                )
                for movie_id in candidate_ids
            ],
            dtype=np.float32,
        )
        final_scores = final_scores + (exploration_weight * candidate_bonus)

    if n >= len(final_scores):
        ranked = final_scores.argsort()[::-1]
    else:
        top = np.argpartition(final_scores, -n)[-n:]
        ranked = top[final_scores[top].argsort()[::-1]]

    return candidate_ids[ranked].tolist()


def _genre_exploration_bonus(
    genres: list[str], genre_impression_counts: dict[str, int]
) -> float:
    """Higher bonus for genres we have shown less often to this user."""
    if not genres:
        return 0.0

    bonuses = [
        1.0 / np.sqrt(1.0 + float(genre_impression_counts.get(genre, 0)))
        for genre in genres
    ]
    return float(max(bonuses, default=0.0))
