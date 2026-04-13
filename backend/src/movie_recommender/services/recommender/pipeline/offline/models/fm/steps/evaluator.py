from __future__ import annotations

import json
import datetime
import pickle
from typing import Dict, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

from movie_recommender.services.recommender.utils.schema import Config
from movie_recommender.services.recommender.pipeline.offline.models.als.steps.metrics import (
    dcg_at_k,
)
from movie_recommender.services.recommender.pipeline.offline.models.fm.steps.data import (
    load_lightfm_data,
)


def run(config: Config) -> dict:
    """Evaluate LightFM on the validation split using Recall@K, Precision@K, NDCG@K."""
    assets_dir = config.data_dirs.model_assets_dir
    k = 10

    train_df = pd.read_parquet(config.data_dirs.splits_dir / "train.parquet")
    val_df = pd.read_parquet(config.data_dirs.splits_dir / "val.parquet")

    train_lookup: Dict[int, Set[int]] = (
        train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    )
    val_lookup: Dict[int, Set[int]] = (
        val_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    )

    with open(assets_dir / "fm_lightfm_model.pkl", "rb") as f:
        model = pickle.load(f)

    interactions, item_features, mappings = load_lightfm_data(config)
    user_id_to_index = {
        int(k_): int(v) for k_, v in mappings["user_id_to_index"].items()
    }
    movie_id_to_index = {
        int(k_): int(v) for k_, v in mappings["movie_id_to_index"].items()
    }

    all_movie_ids = sorted(train_df["movie_id"].unique())
    recall_scores, precision_scores, ndcg_scores = [], [], []

    print("Evaluating LightFM on validation users...")
    for user_id in tqdm(val_lookup.keys()):
        if user_id not in user_id_to_index:
            continue
        seen_movies = train_lookup.get(user_id, set())
        true_movies = val_lookup[user_id]
        if not true_movies:
            continue

        candidate_pairs = [
            (m, movie_id_to_index[m])
            for m in all_movie_ids
            if m not in seen_movies and m in movie_id_to_index
        ]
        if not candidate_pairs:
            continue

        candidate_ids = [m for m, _ in candidate_pairs]
        candidate_indices = np.array(
            [idx for _, idx in candidate_pairs], dtype=np.int32
        )
        scores = model.predict(
            user_ids=user_id_to_index[user_id],
            item_ids=candidate_indices,
            item_features=item_features,
        )

        top_k = np.argpartition(scores, -k)[-k:]
        top_k = top_k[np.argsort(-scores[top_k])]

        recommended = {candidate_ids[i] for i in top_k}
        hits = recommended & true_movies
        relevance = [1 if candidate_ids[i] in true_movies else 0 for i in top_k]
        dcg = dcg_at_k(relevance)
        idcg = dcg_at_k(sorted(relevance, reverse=True))

        recall_scores.append(len(hits) / len(true_movies))
        precision_scores.append(len(hits) / k)
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    print(f"\n=== LightFM Evaluation Results (K={k}) ===")
    if recall_scores:
        print(f"Recall@{k}:    {np.mean(recall_scores):.4f}")
        print(f"Precision@{k}: {np.mean(precision_scores):.4f}")
        print(f"NDCG@{k}:      {np.mean(ndcg_scores):.4f}")
    else:
        print("No users were evaluated.")

    report = {
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": "fm",
        "k": k,
        "config": {
            "min_user_ratings": config.pipeline.min_user_ratings,
            "min_movie_ratings": config.pipeline.min_movie_ratings,
            "train_ratio": config.pipeline.train_ratio,
            "val_ratio": config.pipeline.val_ratio,
        },
        "metrics": {
            f"recall@{k}": float(np.mean(recall_scores)) if recall_scores else 0.0,
            f"precision@{k}": float(np.mean(precision_scores))
            if precision_scores
            else 0.0,
            f"ndcg@{k}": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        },
        "num_users_evaluated": len(recall_scores),
    }

    report_path = assets_dir / "fm_metrics.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Metrics saved to {report_path}")

    return report
