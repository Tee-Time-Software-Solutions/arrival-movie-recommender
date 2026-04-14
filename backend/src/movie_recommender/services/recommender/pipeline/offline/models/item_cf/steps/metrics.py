from __future__ import annotations

import datetime
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from movie_recommender.services.recommender.pipeline.offline.models.item_cf.steps.inference import (
    load_item_cf_artifacts,
    recommend_top_n_for_user,
)
from movie_recommender.services.recommender.utils.schema import Config


def _to_int_mapping(raw_mapping: dict) -> dict[int, int]:
    return {int(key): int(value) for key, value in raw_mapping.items()}


def dcg_at_k(relevance: list[int] | list[float]) -> float:
    return float(sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance)))


def run(config: Config) -> None:
    assets_dir = config.data_dirs.model_assets_dir
    splits_dir = config.data_dirs.splits_dir
    item_cf = config.models.item_cf
    k = 10

    similarity, train_matrix, mappings = load_item_cf_artifacts(config)
    user_id_to_index = _to_int_mapping(mappings["user_id_to_index"])
    movie_id_to_index = _to_int_mapping(mappings["movie_id_to_index"])
    index_to_movie_id = _to_int_mapping(mappings["index_to_movie_id"])

    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df = pd.read_parquet(splits_dir / "val.parquet")
    train_lookup = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    threshold = item_cf.relevance_preference_threshold
    val_lookup = (
        val_df[val_df["preference"] > threshold]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    ndcg_scores: list[float] = []

    print("Evaluating Item-CF on validation users...")
    for user_id in tqdm(val_lookup.keys()):
        if user_id not in user_id_to_index:
            continue

        seen_movies = train_lookup.get(user_id, set())
        candidate_count = sum(
            1 for movie_id in movie_id_to_index if movie_id not in seen_movies
        )
        if candidate_count == 0:
            continue

        true_movies = {
            movie_id
            for movie_id in val_lookup[user_id]
            if movie_id in movie_id_to_index and movie_id not in seen_movies
        }
        if not true_movies:
            continue

        recommendations = recommend_top_n_for_user(
            user_id=user_id,
            n=k,
            similarity=similarity,
            train_matrix=train_matrix,
            user_id_to_index=user_id_to_index,
            movie_id_to_index=movie_id_to_index,
            index_to_movie_id=index_to_movie_id,
            use_positive_only=item_cf.use_positive_only,
            normalize_scores=item_cf.normalize_scores,
            neighbor_weight_power=item_cf.neighbor_weight_power,
            exclude_seen=True,
        )
        if not recommendations:
            continue

        recommended_set = set(recommendations)
        hits = recommended_set & true_movies
        relevance = [
            1 if movie_id in true_movies else 0 for movie_id in recommendations
        ]
        dcg = dcg_at_k(relevance)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = dcg_at_k(ideal_relevance)

        effective_k = max(1, len(recommendations))
        precision_scores.append(len(hits) / effective_k)
        recall_scores.append(len(hits) / len(true_movies))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    precision = float(np.mean(precision_scores)) if precision_scores else 0.0
    recall = float(np.mean(recall_scores)) if recall_scores else 0.0
    ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    print(f"\n=== Item-CF Evaluation Results (K={k}) ===")
    print(f"Precision@{k}: {precision:.4f}")
    print(f"Recall@{k}:    {recall:.4f}")
    print(f"NDCG@{k}:      {ndcg:.4f}")

    report = {
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": "item_cf",
        "k": k,
        "config": {
            "similarity": item_cf.similarity,
            "top_k_neighbors": item_cf.top_k_neighbors,
            "min_similarity": item_cf.min_similarity,
            "use_positive_only": item_cf.use_positive_only,
            "normalize_scores": item_cf.normalize_scores,
            "min_co_raters": item_cf.min_co_raters,
            "similarity_shrinkage": item_cf.similarity_shrinkage,
            "neighbor_weight_power": item_cf.neighbor_weight_power,
            "relevance_preference_threshold": threshold,
            "min_user_ratings": config.pipeline.min_user_ratings,
            "min_movie_ratings": config.pipeline.min_movie_ratings,
            "train_ratio": config.pipeline.train_ratio,
            "val_ratio": config.pipeline.val_ratio,
        },
        "num_users_evaluated": len(precision_scores),
        "metrics": {
            f"precision@{k}": precision,
            f"recall@{k}": recall,
            f"ndcg@{k}": ndcg,
        },
    }
    report_path = assets_dir / "item_cf_metrics.json"
    with open(report_path, "w") as file_obj:
        json.dump(report, file_obj, indent=2)
    print(f"Metrics saved to {report_path}")
