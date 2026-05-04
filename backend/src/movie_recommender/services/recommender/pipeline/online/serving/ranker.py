import numpy as np

from movie_recommender.services.recommender.pipeline.online.artifacts import (
    RecommenderArtifacts,
)
from movie_recommender.services.recommender.pipeline.online.serving.diversity import (
    mmr_rerank,
)


def score_candidates(
    model_artifacts: RecommenderArtifacts,
    user_vector: np.ndarray,
    seen_movie_ids: set[int],
    genre_impression_counts: dict[str, int] | None = None,
    exploration_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score every unseen movie and return (candidate_ids, candidate_embeddings, scores)."""
    all_ids = model_artifacts.all_movie_ids  # shape (N,), int32

    if seen_movie_ids:
        seen = np.fromiter(seen_movie_ids, dtype=np.int32, count=len(seen_movie_ids))
        mask = ~np.isin(all_ids, seen)
    else:
        mask = np.ones(len(all_ids), dtype=bool)

    candidate_ids = all_ids[mask]
    candidate_embeddings = model_artifacts.movie_embeddings[mask]
    scores = candidate_embeddings @ user_vector

    if exploration_weight > 0 and len(candidate_ids) > 0:
        scores = scores.copy()

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

        scores += exploration_weight * bonuses

    return candidate_ids, candidate_embeddings, scores


def select_top_n(
    candidate_ids: np.ndarray,
    candidate_embeddings: np.ndarray,
    scores: np.ndarray,
    n: int,
    diversity_weight: float = 0.0,
) -> list[int]:
    """Pick top-n by score, optionally MMR-reranked for diversity."""
    if n == 0 or len(candidate_ids) == 0:
        return []

    if diversity_weight > 0:
        pool_size = min(len(scores), max(n * 5, n))
        if pool_size < len(scores):
            pool_idx = np.argpartition(scores, -pool_size)[-pool_size:]
        else:
            pool_idx = np.arange(len(scores))
        lambda_mmr = float(np.clip(1.0 - diversity_weight, 0.0, 1.0))
        local = mmr_rerank(
            scores=scores[pool_idx],
            embeddings=candidate_embeddings[pool_idx],
            top_k=n,
            lambda_mmr=lambda_mmr,
        )
        ranked = pool_idx[local]
    elif n >= len(scores):
        ranked = scores.argsort()[::-1]
    else:
        top = np.argpartition(scores, -n)[-n:]
        ranked = top[scores[top].argsort()[::-1]]

    return candidate_ids[ranked].tolist()


def rank_movie_ids(
    n: int,
    model_artifacts: RecommenderArtifacts,
    user_vector: np.ndarray,
    seen_movie_ids: set[int],
    genre_impression_counts: dict[str, int] | None = None,
    exploration_weight: float = 0.0,
    diversity_weight: float = 0.0,
) -> list[int]:
    """Rank all known movies by score, excluding seen ones."""
    candidate_ids, candidate_embeddings, scores = score_candidates(
        model_artifacts=model_artifacts,
        user_vector=user_vector,
        seen_movie_ids=seen_movie_ids,
        genre_impression_counts=genre_impression_counts,
        exploration_weight=exploration_weight,
    )
    return select_top_n(
        candidate_ids=candidate_ids,
        candidate_embeddings=candidate_embeddings,
        scores=scores,
        n=n,
        diversity_weight=diversity_weight,
    )
