from __future__ import annotations

import json
import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from movie_recommender.services.recommender.utils.schema import Config


def dcg_at_k(relevance: list[int] | list[float]) -> float:
    """Discounted Cumulative Gain at K for a ranked relevance list."""
    return float(sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance)))


def run(config: Config) -> None:
    assets_dir = config.data_dirs.model_assets_dir
    splits_dir = config.data_dirs.splits_dir
    k = 10

    print("Loading embeddings...")
    user_embeddings = np.load(assets_dir / "user_embeddings.npy")
    movie_embeddings = np.load(assets_dir / "movie_embeddings.npy")

    with open(assets_dir / "mappings.json") as f:
        mappings = json.load(f)

    user_id_to_index = {int(k_): v for k_, v in mappings["user_id_to_index"].items()}
    movie_id_to_index = {int(k_): v for k_, v in mappings["movie_id_to_index"].items()}
    index_to_movie_id = {int(k_): v for k_, v in mappings["index_to_movie_id"].items()}

    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df = pd.read_parquet(splits_dir / "val.parquet")

    train_lookup = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    val_lookup = val_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    recall_scores, precision_scores, ndcg_scores = [], [], []

    print("Evaluating...")
    for user_id in tqdm(val_lookup.keys()):
        if user_id not in user_id_to_index:
            continue

        user_vector = user_embeddings[user_id_to_index[user_id]]
        scores = movie_embeddings @ user_vector

        seen_indices = [
            movie_id_to_index[m]
            for m in train_lookup.get(user_id, set())
            if m in movie_id_to_index
        ]
        scores[seen_indices] = -np.inf

        top_k = np.argpartition(scores, -k)[-k:]
        top_k = top_k[np.argsort(-scores[top_k])]

        recommended = {index_to_movie_id[idx] for idx in top_k}
        true_movies = val_lookup[user_id]
        hits = recommended & true_movies

        relevance = [1 if index_to_movie_id[idx] in true_movies else 0 for idx in top_k]
        dcg = dcg_at_k(relevance)
        idcg = dcg_at_k(sorted(relevance, reverse=True))

        recall_scores.append(len(hits) / len(true_movies))
        precision_scores.append(len(hits) / k)
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0)

    print(f"\n=== ALS Evaluation Results (K={k}) ===")
    print(f"Recall@{k}:    {np.mean(recall_scores):.4f}")
    print(f"Precision@{k}: {np.mean(precision_scores):.4f}")
    print(f"NDCG@{k}:      {np.mean(ndcg_scores):.4f}")

    report = {
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": "als",
        "k": k,
        "config": {
            "factors": config.models.als.factors,
            "regularization": config.models.als.regularization,
            "iterations": config.models.als.iterations,
            "alpha": config.models.als.alpha,
            "min_user_ratings": config.pipeline.min_user_ratings,
            "min_movie_ratings": config.pipeline.min_movie_ratings,
            "train_ratio": config.pipeline.train_ratio,
            "val_ratio": config.pipeline.val_ratio,
        },
        "metrics": {
            f"recall@{k}": float(np.mean(recall_scores)),
            f"precision@{k}": float(np.mean(precision_scores)),
            f"ndcg@{k}": float(np.mean(ndcg_scores)),
        },
        "num_users_evaluated": len(recall_scores),
    }

    report_path = assets_dir / "als_metrics.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Metrics saved to {report_path}")
