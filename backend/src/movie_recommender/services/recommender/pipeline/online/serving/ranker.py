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
    final_scores = scores

    if exploration_weight > 0:
        final_scores = scores.copy()

        counts = genre_impression_counts or {}
        genre_to_bonus = {g: 1.0 / np.sqrt(1.0 + c) for g, c in counts.items()}

        movie_to_genres = model_artifacts.movie_id_to_genres
        bonuses = np.zeros(len(candidate_ids), dtype=np.float32)

        for i, movie_id in enumerate(candidate_ids):
            m_genres = movie_to_genres.get(int(movie_id))
            if m_genres:
                bonuses[i] = max(
                    (genre_to_bonus.get(g, 1.0) for g in m_genres),
                    default=0.0,
                )

        final_scores += exploration_weight * bonuses

    if n >= len(final_scores):
        ranked = final_scores.argsort()[::-1]
    else:
        top = np.argpartition(final_scores, -n)[-n:]
        ranked = top[final_scores[top].argsort()[::-1]]

    return candidate_ids[ranked].tolist()
