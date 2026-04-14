import numpy as np

from movie_recommender.services.recommender.pipeline.online.artifacts import (
    RecommenderArtifacts,
)


def rank_movie_ids(
    n: int,
    model_artifacts: RecommenderArtifacts,
    user_vector: np.ndarray,
    seen_movie_ids: set[int],
) -> list[int]:
    """Rank all known movies by dot-product score, excluding seen ones.

    Fully vectorised: uses the pre-built all_movie_ids array so there are no
    Python-level per-movie loops or dict lookups on the hot path.
    """
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

    if n >= len(scores):
        ranked = scores.argsort()[::-1]
    else:
        # argpartition is O(N) vs argsort's O(N log N) — significant for large n
        top = np.argpartition(scores, -n)[-n:]
        ranked = top[scores[top].argsort()[::-1]]

    return candidate_ids[ranked].tolist()
