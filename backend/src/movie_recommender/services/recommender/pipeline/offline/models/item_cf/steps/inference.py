from __future__ import annotations

import json

import numpy as np
from scipy.sparse import csr_matrix, load_npz

from movie_recommender.services.recommender.utils.schema import Config


def load_item_cf_artifacts(config: Config) -> tuple[csr_matrix, csr_matrix, dict]:
    assets_dir = config.data_dirs.model_assets_dir
    similarity = load_npz(assets_dir / "item_cf_similarity.npz").tocsr()
    train_matrix = load_npz(assets_dir / "item_cf_train_matrix.npz").tocsr()
    with open(assets_dir / "item_cf_mappings.json") as file_obj:
        mappings = json.load(file_obj)
    return similarity, train_matrix, mappings


def _to_int_mapping(raw_mapping: dict) -> dict[int, int]:
    return {int(key): int(value) for key, value in raw_mapping.items()}


def _get_user_history(
    user_idx: int,
    train_matrix: csr_matrix,
    use_positive_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    user_row = train_matrix.getrow(user_idx)
    history_indices = user_row.indices
    history_values = user_row.data.astype(np.float32)

    if use_positive_only:
        positive_mask = history_values > 0.0
        history_indices = history_indices[positive_mask]
        history_values = history_values[positive_mask]

    return history_indices, history_values


def _apply_neighbor_weight_power(
    weights: np.ndarray, weight_power: float
) -> np.ndarray:
    if weight_power <= 0:
        raise ValueError("neighbor_weight_power must be greater than 0")
    if weight_power == 1.0:
        return weights
    return np.sign(weights) * np.power(np.abs(weights), weight_power)


def score_user_movie(
    user_id: int,
    movie_id: int,
    similarity: csr_matrix,
    train_matrix: csr_matrix,
    user_id_to_index: dict[int, int],
    movie_id_to_index: dict[int, int],
    use_positive_only: bool = True,
    normalize_scores: bool = True,
    neighbor_weight_power: float = 1.0,
) -> float:
    user_idx = user_id_to_index.get(user_id)
    item_idx = movie_id_to_index.get(movie_id)
    if user_idx is None or item_idx is None:
        return 0.0

    history_indices, history_values = _get_user_history(
        user_idx=user_idx,
        train_matrix=train_matrix,
        use_positive_only=use_positive_only,
    )
    if history_indices.size == 0:
        return 0.0

    similarity_row = similarity.getrow(item_idx)
    if similarity_row.nnz == 0:
        return 0.0

    lookup = {idx: value for idx, value in zip(history_indices, history_values)}
    weights = []
    preferences = []
    for neighbor_idx, weight in zip(similarity_row.indices, similarity_row.data):
        preference = lookup.get(neighbor_idx)
        if preference is None:
            continue
        weights.append(float(weight))
        preferences.append(float(preference))

    if not weights:
        return 0.0

    weight_array = np.array(weights, dtype=np.float32)
    weight_array = _apply_neighbor_weight_power(
        weights=weight_array,
        weight_power=neighbor_weight_power,
    )
    pref_array = np.array(preferences, dtype=np.float32)
    score = float(np.dot(weight_array, pref_array))
    if not normalize_scores:
        return score

    denominator = float(np.sum(np.abs(weight_array)))
    if denominator <= 0.0:
        return 0.0
    return score / denominator


def recommend_top_n_for_user(
    user_id: int,
    n: int,
    similarity: csr_matrix,
    train_matrix: csr_matrix,
    user_id_to_index: dict[int, int],
    movie_id_to_index: dict[int, int],
    index_to_movie_id: dict[int, int],
    use_positive_only: bool = True,
    normalize_scores: bool = True,
    neighbor_weight_power: float = 1.0,
    exclude_seen: bool = True,
) -> list[int]:
    user_idx = user_id_to_index.get(user_id)
    if user_idx is None or n <= 0:
        return []

    seen_item_indices = set(train_matrix.getrow(user_idx).indices.tolist())
    scores: list[tuple[float, int]] = []
    for item_idx, movie_id in index_to_movie_id.items():
        if exclude_seen and item_idx in seen_item_indices:
            continue
        score = score_user_movie(
            user_id=user_id,
            movie_id=movie_id,
            similarity=similarity,
            train_matrix=train_matrix,
            user_id_to_index=user_id_to_index,
            movie_id_to_index=movie_id_to_index,
            use_positive_only=use_positive_only,
            normalize_scores=normalize_scores,
            neighbor_weight_power=neighbor_weight_power,
        )
        scores.append((score, movie_id))

    if not scores:
        return []

    scores.sort(key=lambda row: row[0], reverse=True)
    return [movie_id for _, movie_id in scores[:n]]


def recommend_top_n_from_artifacts(
    config: Config,
    user_id: int,
    n: int = 10,
) -> list[int]:
    similarity, train_matrix, mappings = load_item_cf_artifacts(config)
    user_id_to_index = _to_int_mapping(mappings["user_id_to_index"])
    movie_id_to_index = _to_int_mapping(mappings["movie_id_to_index"])
    index_to_movie_id = _to_int_mapping(mappings["index_to_movie_id"])

    return recommend_top_n_for_user(
        user_id=user_id,
        n=n,
        similarity=similarity,
        train_matrix=train_matrix,
        user_id_to_index=user_id_to_index,
        movie_id_to_index=movie_id_to_index,
        index_to_movie_id=index_to_movie_id,
        use_positive_only=config.models.item_cf.use_positive_only,
        normalize_scores=config.models.item_cf.normalize_scores,
        neighbor_weight_power=config.models.item_cf.neighbor_weight_power,
        exclude_seen=True,
    )
