from __future__ import annotations

import json
from typing import Dict, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

from movie_recommender.services.recommender.paths_dev import DATA_SPLITS, artifacts_dir
from movie_recommender.services.recommender.learning.metrics import dcg_at_k
from movie_recommender.services.recommender.learning.fm.inference import (
    _load_lightfm_model,
    _load_item_features,
    _load_mappings,
)


TRAIN_PATH = DATA_SPLITS / "train.parquet"
VAL_PATH = DATA_SPLITS / "val.parquet"

K = 10


def _calculate_metrics(
    *,
    recommended_ids: set[int],
    true_ids: set[int],
    relevance_list: list[int],
    k: int,
) -> tuple[float, float, float]:
    hits = recommended_ids & true_ids

    recall = 0.0 if not true_ids else len(hits) / len(true_ids)
    precision = 0.0 if k <= 0 else len(hits) / k

    dcg = dcg_at_k(relevance_list)
    idcg = dcg_at_k(sorted(relevance_list, reverse=True))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return recall, precision, ndcg


def compare_als_vs_fm() -> None:
    """
    Evaluate ALS and LightFM-FM on the same validation users and print metrics side by side.
    """
    print("Loading splits...")
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)

    print("Building lookup tables...")
    train_lookup: Dict[int, Set[int]] = (
        train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    )
    val_lookup: Dict[int, Set[int]] = (
        val_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    )

    print("Loading ALS artifacts...")
    als_root = artifacts_dir()
    user_embeddings = np.load(als_root / "user_embeddings.npy")
    movie_embeddings = np.load(als_root / "movie_embeddings.npy")

    with open(als_root / "mappings.json", "r") as f:
        mappings = json.load(f)

    user_id_to_index = {int(k): v for k, v in mappings["user_id_to_index"].items()}
    movie_id_to_index = {int(k): v for k, v in mappings["movie_id_to_index"].items()}
    index_to_movie_id = {int(k): v for k, v in mappings["index_to_movie_id"].items()}

    # Load LightFM artifacts once
    print("Loading LightFM artifacts...")
    fm_model = _load_lightfm_model()
    fm_item_features = _load_item_features()
    fm_mappings = _load_mappings()
    fm_user_id_to_index = {
        int(k): int(v) for k, v in fm_mappings["user_id_to_index"].items()
    }
    fm_movie_id_to_index = {
        int(k): int(v) for k, v in fm_mappings["movie_id_to_index"].items()
    }

    all_movie_ids = sorted(train_df["movie_id"].unique())

    als_recall_scores = []
    als_precision_scores = []
    als_ndcg_scores = []

    fm_recall_scores = []
    fm_precision_scores = []
    fm_ndcg_scores = []

    als_users = set(user_id_to_index.keys())
    fm_users = set(fm_user_id_to_index.keys())
    val_users = {u for u, true_movies in val_lookup.items() if true_movies}
    comparison_users = val_users & als_users & fm_users

    print(
        "Comparing ALS and LightFM on validation users (vectorized FM)...\n"
        f"Users in validation with ground truth: {len(val_users)}\n"
        f"Users present in both models: {len(comparison_users)}"
    )

    for user_id in tqdm(sorted(comparison_users)):
        true_movies = val_lookup[user_id]

        seen_movies = train_lookup.get(user_id, set())

        # ---------- ALS ----------
        user_index = user_id_to_index[user_id]
        user_vector = user_embeddings[user_index]

        als_scores = movie_embeddings @ user_vector

        seen_indices = [
            movie_id_to_index[m] for m in seen_movies if m in movie_id_to_index
        ]
        als_scores[seen_indices] = -np.inf

        k_eff_als = min(K, int(als_scores.shape[0]))
        if k_eff_als <= 0:
            continue

        top_k_indices = np.argpartition(als_scores, -k_eff_als)[-k_eff_als:]
        top_k_indices = top_k_indices[np.argsort(-als_scores[top_k_indices])]

        als_recommended = {index_to_movie_id[idx] for idx in top_k_indices}

        als_relevance = [
            1 if index_to_movie_id[idx] in true_movies else 0 for idx in top_k_indices
        ]
        als_recall, als_precision, als_ndcg = _calculate_metrics(
            recommended_ids=als_recommended,
            true_ids=true_movies,
            relevance_list=als_relevance,
            k=k_eff_als,
        )

        als_recall_scores.append(als_recall)
        als_precision_scores.append(als_precision)
        als_ndcg_scores.append(als_ndcg)

        # ---------- LightFM FM ----------
        candidate_pairs = [
            (m, fm_movie_id_to_index[m])
            for m in all_movie_ids
            if m not in seen_movies and m in fm_movie_id_to_index
        ]
        if not candidate_pairs:
            continue

        candidate_movie_ids = [m for m, _ in candidate_pairs]
        candidate_item_indices = np.array(
            [idx for _, idx in candidate_pairs], dtype=np.int32
        )

        fm_u_idx = fm_user_id_to_index[user_id]

        fm_scores = fm_model.predict(
            user_ids=fm_u_idx,
            item_ids=candidate_item_indices,
            item_features=fm_item_features,
        )

        k_eff_fm = min(K, int(fm_scores.shape[0]))
        if k_eff_fm <= 0:
            continue

        fm_top_k_indices = np.argpartition(fm_scores, -k_eff_fm)[-k_eff_fm:]
        fm_top_k_indices = fm_top_k_indices[
            np.argsort(-fm_scores[fm_top_k_indices])
        ]

        fm_recommended = {candidate_movie_ids[i] for i in fm_top_k_indices}

        fm_relevance = [
            1 if candidate_movie_ids[i] in true_movies else 0
            for i in fm_top_k_indices
        ]
        fm_recall, fm_precision, fm_ndcg = _calculate_metrics(
            recommended_ids=fm_recommended,
            true_ids=true_movies,
            relevance_list=fm_relevance,
            k=k_eff_fm,
        )

        fm_recall_scores.append(fm_recall)
        fm_precision_scores.append(fm_precision)
        fm_ndcg_scores.append(fm_ndcg)

    print("\n=== ALS vs LightFM Comparison (Validation) ===")
    print(f"Users evaluated: {len(als_recall_scores)}")
    print("Metric        ALS         LightFM")
    print(
        f"Recall@{K}:   {np.mean(als_recall_scores):.4f}    {np.mean(fm_recall_scores):.4f}"
    )
    print(
        f"Precision@{K}:{np.mean(als_precision_scores):.4f}    {np.mean(fm_precision_scores):.4f}"
    )
    print(
        f"NDCG@{K}:     {np.mean(als_ndcg_scores):.4f}    {np.mean(fm_ndcg_scores):.4f}"
    )


if __name__ == "__main__":
    compare_als_vs_fm()

