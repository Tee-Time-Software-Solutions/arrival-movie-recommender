from __future__ import annotations

import json
from typing import Dict, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

from movie_recommender.services.recommender.paths_dev import ARTIFACTS, DATA_SPLITS
from movie_recommender.services.recommender.learning.fm.inference import (
    _load_lightfm_model,
    _load_item_features,
    _load_mappings,
)


TRAIN_PATH = DATA_SPLITS / "train.parquet"
VAL_PATH = DATA_SPLITS / "val.parquet"

ALS_USER_EMB_PATH = ARTIFACTS / "user_embeddings.npy"
ALS_MOVIE_EMB_PATH = ARTIFACTS / "movie_embeddings.npy"
MAPPINGS_PATH = ARTIFACTS / "mappings.json"

K = 10


def dcg_at_k(relevance):
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance))


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
    user_embeddings = np.load(ALS_USER_EMB_PATH)
    movie_embeddings = np.load(ALS_MOVIE_EMB_PATH)

    with open(MAPPINGS_PATH, "r") as f:
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

    print("Comparing ALS and LightFM on validation users (vectorized FM)...")
    for user_id in tqdm(val_lookup.keys()):
        true_movies = val_lookup[user_id]
        if not true_movies:
            continue

        seen_movies = train_lookup.get(user_id, set())

        # ---------- ALS ----------
        if user_id not in user_id_to_index:
            # User not present in ALS training set
            continue

        user_index = user_id_to_index[user_id]
        user_vector = user_embeddings[user_index]

        als_scores = movie_embeddings @ user_vector

        seen_indices = [
            movie_id_to_index[m] for m in seen_movies if m in movie_id_to_index
        ]
        als_scores[seen_indices] = -np.inf

        top_k_indices = np.argpartition(als_scores, -K)[-K:]
        top_k_indices = top_k_indices[np.argsort(-als_scores[top_k_indices])]

        als_recommended = {index_to_movie_id[idx] for idx in top_k_indices}

        als_hits = als_recommended & true_movies

        als_recall = len(als_hits) / len(true_movies)
        als_precision = len(als_hits) / K

        als_relevance = [
            1 if index_to_movie_id[idx] in true_movies else 0 for idx in top_k_indices
        ]
        als_dcg = dcg_at_k(als_relevance)
        als_idcg = dcg_at_k(sorted(als_relevance, reverse=True))
        als_ndcg = als_dcg / als_idcg if als_idcg > 0 else 0.0

        als_recall_scores.append(als_recall)
        als_precision_scores.append(als_precision)
        als_ndcg_scores.append(als_ndcg)

        # ---------- LightFM FM ----------
        if user_id not in fm_user_id_to_index:
            # User not present in LightFM training set
            continue

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

        fm_top_k_indices = np.argpartition(fm_scores, -K)[-K:]
        fm_top_k_indices = fm_top_k_indices[
            np.argsort(-fm_scores[fm_top_k_indices])
        ]

        fm_recommended = {candidate_movie_ids[i] for i in fm_top_k_indices}
        fm_hits = fm_recommended & true_movies

        fm_recall = len(fm_hits) / len(true_movies)
        fm_precision = len(fm_hits) / K

        fm_relevance = [
            1 if candidate_movie_ids[i] in true_movies else 0
            for i in fm_top_k_indices
        ]
        fm_dcg = dcg_at_k(fm_relevance)
        fm_idcg = dcg_at_k(sorted(fm_relevance, reverse=True))
        fm_ndcg = fm_dcg / fm_idcg if fm_idcg > 0 else 0.0

        fm_recall_scores.append(fm_recall)
        fm_precision_scores.append(fm_precision)
        fm_ndcg_scores.append(fm_ndcg)

    print("\n=== ALS vs LightFM Comparison (Validation) ===")
    print(f"Users evaluated: {len(als_recall_scores)}")
    print(f"Metric        ALS         LightFM")
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

