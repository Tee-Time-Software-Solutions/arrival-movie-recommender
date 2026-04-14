from __future__ import annotations

import datetime
import json

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, diags, load_npz, save_npz

from movie_recommender.services.recommender.utils.schema import Config


def _filter_training_matrix(
    interaction_matrix: csr_matrix, use_positive_only: bool
) -> csr_matrix:
    filtered = interaction_matrix.copy().astype(np.float32)
    if use_positive_only:
        filtered.data = np.where(filtered.data > 0.0, filtered.data, 0.0)
        filtered.eliminate_zeros()
    return filtered


def _compute_cosine_similarity(filtered_matrix: csr_matrix) -> csr_matrix:
    item_matrix: csc_matrix = filtered_matrix.tocsc(copy=False)
    item_norms = np.sqrt(item_matrix.power(2).sum(axis=0)).A1
    safe_norms = np.where(item_norms > 0.0, item_norms, 1.0)
    inverse_norms = 1.0 / safe_norms

    normalized_matrix = item_matrix @ diags(inverse_norms.astype(np.float32))
    similarity = (normalized_matrix.T @ normalized_matrix).tocsr().astype(np.float32)
    similarity.setdiag(0.0)
    similarity.eliminate_zeros()
    return similarity


def _compute_co_rater_counts(filtered_matrix: csr_matrix) -> csr_matrix:
    binary = filtered_matrix.copy().astype(np.float32)
    binary.data = np.ones_like(binary.data, dtype=np.float32)
    return (binary.T @ binary).tocsr().astype(np.float32)


def _apply_co_rater_controls(
    similarity: csr_matrix,
    co_rater_counts: csr_matrix,
    min_co_raters: int,
    similarity_shrinkage: float,
) -> csr_matrix:
    if min_co_raters <= 0:
        raise ValueError("min_co_raters must be greater than 0")
    if similarity_shrinkage < 0:
        raise ValueError("similarity_shrinkage must be non-negative")

    updated_similarity = similarity

    if min_co_raters > 1:
        co_mask = co_rater_counts.copy()
        co_mask.data = (co_mask.data >= min_co_raters).astype(np.float32)
        co_mask.eliminate_zeros()
        updated_similarity = updated_similarity.multiply(co_mask)
        updated_similarity.eliminate_zeros()

    if similarity_shrinkage > 0:
        shrinkage = co_rater_counts.copy().astype(np.float32)
        shrinkage.data = shrinkage.data / (shrinkage.data + similarity_shrinkage)
        updated_similarity = updated_similarity.multiply(shrinkage)
        updated_similarity.eliminate_zeros()

    return updated_similarity


def _prune_top_k_neighbors(
    similarity: csr_matrix, top_k_neighbors: int, min_similarity: float
) -> csr_matrix:
    if top_k_neighbors <= 0:
        raise ValueError("top_k_neighbors must be greater than 0")

    pruned_data: list[np.ndarray] = []
    pruned_indices: list[np.ndarray] = []
    indptr = [0]

    for row_index in range(similarity.shape[0]):
        row = similarity.getrow(row_index)
        if row.nnz == 0:
            indptr.append(indptr[-1])
            continue

        values = row.data
        indices = row.indices
        keep_mask = values >= min_similarity
        values = values[keep_mask]
        indices = indices[keep_mask]

        if values.size == 0:
            indptr.append(indptr[-1])
            continue

        if values.size > top_k_neighbors:
            top_positions = np.argpartition(values, -top_k_neighbors)[-top_k_neighbors:]
            values = values[top_positions]
            indices = indices[top_positions]

        order = np.argsort(indices)
        values = values[order]
        indices = indices[order]

        pruned_data.append(values.astype(np.float32))
        pruned_indices.append(indices.astype(np.int32))
        indptr.append(indptr[-1] + values.size)

    if pruned_data:
        data = np.concatenate(pruned_data)
        indices = np.concatenate(pruned_indices)
    else:
        data = np.array([], dtype=np.float32)
        indices = np.array([], dtype=np.int32)

    return csr_matrix(
        (data, indices, np.array(indptr, dtype=np.int32)),
        shape=similarity.shape,
        dtype=np.float32,
    )


def run(config: Config) -> None:
    assets_dir = config.data_dirs.model_assets_dir
    item_cf = config.models.item_cf

    if item_cf.similarity != "cosine":
        raise ValueError(f"Unsupported item_cf similarity metric: {item_cf.similarity}")

    print("Loading Item-CF train matrix...")
    interaction_matrix = load_npz(assets_dir / "item_cf_train_matrix.npz").tocsr()

    filtered_matrix = _filter_training_matrix(
        interaction_matrix=interaction_matrix,
        use_positive_only=item_cf.use_positive_only,
    )
    print(
        "Building item-item cosine similarity "
        f"(use_positive_only={item_cf.use_positive_only})..."
    )
    similarity = _compute_cosine_similarity(filtered_matrix=filtered_matrix)
    co_rater_counts = _compute_co_rater_counts(filtered_matrix=filtered_matrix)
    similarity = _apply_co_rater_controls(
        similarity=similarity,
        co_rater_counts=co_rater_counts,
        min_co_raters=item_cf.min_co_raters,
        similarity_shrinkage=item_cf.similarity_shrinkage,
    )
    similarity = _prune_top_k_neighbors(
        similarity=similarity,
        top_k_neighbors=item_cf.top_k_neighbors,
        min_similarity=item_cf.min_similarity,
    )

    save_npz(assets_dir / "item_cf_similarity.npz", similarity)

    with open(assets_dir / "item_cf_model_info.json", "w") as file_obj:
        json.dump(
            {
                "evaluated_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "model": "item_cf",
                "similarity": item_cf.similarity,
                "top_k_neighbors": item_cf.top_k_neighbors,
                "min_similarity": item_cf.min_similarity,
                "use_positive_only": item_cf.use_positive_only,
                "normalize_scores": item_cf.normalize_scores,
                "min_co_raters": item_cf.min_co_raters,
                "similarity_shrinkage": item_cf.similarity_shrinkage,
                "num_users": int(interaction_matrix.shape[0]),
                "num_items": int(interaction_matrix.shape[1]),
                "similarity_nnz": int(similarity.nnz),
            },
            file_obj,
            indent=2,
        )

    print(
        "Item-CF similarity training complete. "
        f"Similarity shape: {similarity.shape}, nnz={similarity.nnz}"
    )
