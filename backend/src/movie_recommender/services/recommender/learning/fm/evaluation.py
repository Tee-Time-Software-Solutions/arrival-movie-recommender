from __future__ import annotations

from typing import Dict, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

from movie_recommender.services.recommender.paths_dev import DATA_SPLITS
from movie_recommender.services.recommender.learning.metrics import dcg_at_k
from movie_recommender.services.recommender.learning.fm.inference import (
    _load_lightfm_model,
    _load_item_features,
    _load_mappings,
)


TRAIN_PATH = DATA_SPLITS / "train.parquet"
VAL_PATH = DATA_SPLITS / "val.parquet"

K = 10


def evaluate_fm() -> None:
    """
    Evaluate the LightFM model on the validation split using Recall@K, Precision@K and NDCG@K.
    Uses vectorized prediction per user for efficiency while scoring all candidate items.
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

    # Load LightFM artifacts once
    print("Loading LightFM model and feature matrices...")
    model = _load_lightfm_model()
    item_features = _load_item_features()
    mappings = _load_mappings()

    user_id_to_index = {int(k): int(v) for k, v in mappings["user_id_to_index"].items()}
    movie_id_to_index = {
        int(k): int(v) for k, v in mappings["movie_id_to_index"].items()
    }

    all_movie_ids = sorted(train_df["movie_id"].unique())

    recall_scores = []
    precision_scores = []
    ndcg_scores = []

    print("Evaluating LightFM on validation users (vectorized per user)...")
    for user_id in tqdm(val_lookup.keys()):
        # Skip users not seen in training for LightFM
        if user_id not in user_id_to_index:
            continue

        seen_movies = train_lookup.get(user_id, set())
        true_movies = val_lookup[user_id]
        if not true_movies:
            continue

        # Build candidate movie IDs and corresponding LightFM item indices
        candidate_pairs = [
            (m, movie_id_to_index[m])
            for m in all_movie_ids
            if m not in seen_movies and m in movie_id_to_index
        ]
        if not candidate_pairs:
            continue

        candidate_movie_ids = [m for m, _ in candidate_pairs]
        candidate_item_indices = np.array(
            [idx for _, idx in candidate_pairs], dtype=np.int32
        )

        u_idx = user_id_to_index[user_id]

        # Vectorized prediction over all candidate items
        scores = model.predict(
            user_ids=u_idx,
            item_ids=candidate_item_indices,
            item_features=item_features,
        )

        top_k_indices = np.argpartition(scores, -K)[-K:]
        top_k_indices = top_k_indices[np.argsort(-scores[top_k_indices])]

        recommended_movies = {candidate_movie_ids[i] for i in top_k_indices}

        hits = recommended_movies & true_movies

        recall = len(hits) / len(true_movies)
        precision = len(hits) / K

        relevance = [
            1 if candidate_movie_ids[i] in true_movies else 0 for i in top_k_indices
        ]
        dcg = dcg_at_k(relevance)
        idcg = dcg_at_k(sorted(relevance, reverse=True))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        recall_scores.append(recall)
        precision_scores.append(precision)
        ndcg_scores.append(ndcg)

    print("\n=== LightFM Evaluation Results ===")
    if recall_scores:
        print(f"Recall@{K}:    {np.mean(recall_scores):.4f}")
        print(f"Precision@{K}: {np.mean(precision_scores):.4f}")
        print(f"NDCG@{K}:      {np.mean(ndcg_scores):.4f}")
    else:
        print("No users were evaluated (empty recall list).")


if __name__ == "__main__":
    evaluate_fm()
