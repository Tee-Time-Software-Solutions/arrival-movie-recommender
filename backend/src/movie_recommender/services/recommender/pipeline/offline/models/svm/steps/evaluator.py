from __future__ import annotations

import datetime
import json

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from movie_recommender.services.recommender.pipeline.offline.models.als.steps.metrics import (
    dcg_at_k,
)
from movie_recommender.services.recommender.pipeline.offline.models.svm.steps.data import (
    build_features_for_user_candidates,
    load_svm_training_data,
)
from movie_recommender.services.recommender.pipeline.offline.models.svm.steps.trainer import (
    SVM_MODEL_FILENAME,
)
from movie_recommender.services.recommender.utils.schema import Config


def run(config: Config) -> None:
    """Evaluate SVM with ranking metrics on the validation split."""
    assets_dir = config.data_dirs.model_assets_dir
    k = 10

    train_df = pd.read_parquet(config.data_dirs.splits_dir / "train.parquet")
    val_df = pd.read_parquet(config.data_dirs.splits_dir / "val.parquet")

    train_lookup = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    val_positive_df = val_df[val_df["preference"] > 0].copy()
    val_lookup = val_positive_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    model = joblib.load(assets_dir / SVM_MODEL_FILENAME)
    _, _, mappings = load_svm_training_data(config)
    all_movie_ids = sorted(mappings["movie_id_to_index"].keys())

    recall_scores: list[float] = []
    precision_scores: list[float] = []
    ndcg_scores: list[float] = []

    print("Evaluating SVM on validation users...")
    for user_id in tqdm(val_lookup.keys()):
        seen_movies = train_lookup.get(user_id, set())
        true_movies = val_lookup[user_id]
        if not true_movies:
            continue

        candidate_movie_ids = [
            movie_id for movie_id in all_movie_ids if movie_id not in seen_movies
        ]
        if not candidate_movie_ids:
            continue

        features, valid_movies = build_features_for_user_candidates(
            user_id=int(user_id),
            candidate_movie_ids=candidate_movie_ids,
            mappings=mappings,
        )
        if features.shape[0] == 0:
            continue

        scores = model.decision_function(features)
        effective_k = min(k, len(valid_movies))
        top_k = np.argpartition(scores, -effective_k)[-effective_k:]
        top_k = top_k[np.argsort(-scores[top_k])]

        recommended = {valid_movies[idx] for idx in top_k}
        hits = recommended & true_movies
        relevance = [1 if valid_movies[idx] in true_movies else 0 for idx in top_k]
        dcg = dcg_at_k(relevance)
        idcg = dcg_at_k(sorted(relevance, reverse=True))

        recall_scores.append(len(hits) / len(true_movies))
        precision_scores.append(len(hits) / k)
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    recall = float(np.mean(recall_scores)) if recall_scores else 0.0
    precision = float(np.mean(precision_scores)) if precision_scores else 0.0
    ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    print(f"\n=== SVM Evaluation Results (K={k}) ===")
    print(f"Recall@{k}:    {recall:.4f}")
    print(f"Precision@{k}: {precision:.4f}")
    print(f"NDCG@{k}:      {ndcg:.4f}")

    svm_cfg = config.models.svm
    report = {
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": "svm",
        "k": k,
        "config": {
            "c": svm_cfg.c,
            "max_iter": svm_cfg.max_iter,
            "negative_sampling_ratio": svm_cfg.negative_sampling_ratio,
            "random_state": svm_cfg.random_state,
            "use_metadata_features": svm_cfg.use_metadata_features,
            "release_year_bucket_size": svm_cfg.release_year_bucket_size,
            "min_user_ratings": config.pipeline.min_user_ratings,
            "min_movie_ratings": config.pipeline.min_movie_ratings,
            "train_ratio": config.pipeline.train_ratio,
            "val_ratio": config.pipeline.val_ratio,
        },
        "metrics": {
            f"recall@{k}": recall,
            f"precision@{k}": precision,
            f"ndcg@{k}": ndcg,
        },
        "num_users_evaluated": len(recall_scores),
    }

    report_path = assets_dir / "svm_metrics.json"
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"Metrics saved to {report_path}")
