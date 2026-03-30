from typing import Iterable

import numpy as np

from movie_recommender.services.recommender.serving.artifact_loader import (
    RecommenderArtifacts,
)


def top_n_indices(scores: np.ndarray, n: int) -> list[int]:
    if n <= 0 or len(scores) == 0:
        return []

    n = min(n, len(scores))
    if n == len(scores):
        return scores.argsort()[::-1].tolist()

    indices = scores.argpartition(-n)[-n:]
    return indices[scores[indices].argsort()[::-1]].tolist()


def rank_movie_ids(
    artifacts: RecommenderArtifacts,
    user_vector: np.ndarray,
    movie_ids: Iterable[int],
    seen_movie_ids: set[int],
) -> list[int]:
    candidate_movie_ids: list[int] = []
    candidate_indices: list[int] = []
    unknown_movie_ids: list[int] = []

    for movie_id in movie_ids:
        normalized_movie_id = int(movie_id)
        if normalized_movie_id in seen_movie_ids:
            continue

        movie_index = artifacts.movie_id_to_index.get(normalized_movie_id)
        if movie_index is None:
            unknown_movie_ids.append(normalized_movie_id)
            continue

        candidate_movie_ids.append(normalized_movie_id)
        candidate_indices.append(movie_index)

    if not candidate_movie_ids:
        return unknown_movie_ids

    scores = artifacts.movie_embeddings[candidate_indices] @ user_vector
    ranked_local_indices = top_n_indices(scores, len(candidate_movie_ids))
    ranked_ids = [candidate_movie_ids[index] for index in ranked_local_indices]
    ranked_ids.extend(unknown_movie_ids)
    return ranked_ids
