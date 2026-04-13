from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.svm import LinearSVC

from movie_recommender.services.recommender.pipeline.offline.models.svm.steps.data import (
    load_svm_training_data,
)
from movie_recommender.services.recommender.utils.schema import Config

SVM_MODEL_FILENAME = "svm_linear_model.joblib"
SVM_MODEL_INFO_FILENAME = "svm_model_info.json"


def run(config: Config) -> None:
    """Train and persist a sparse linear SVM baseline."""
    assets_dir = config.data_dirs.model_assets_dir
    svm_cfg = config.models.svm

    features, labels, mappings = load_svm_training_data(config)
    print(
        f"SVM train matrix: samples={features.shape[0]}, features={features.shape[1]}"
    )

    model = LinearSVC(
        C=svm_cfg.c,
        max_iter=svm_cfg.max_iter,
        random_state=svm_cfg.random_state,
    )
    model.fit(features, labels)

    joblib.dump(model, assets_dir / SVM_MODEL_FILENAME)

    model_info = {
        "model": "linear_svm",
        "num_samples": int(features.shape[0]),
        "num_features": int(features.shape[1]),
        "num_users": int(len(mappings["user_id_to_index"])),
        "num_movies": int(len(mappings["movie_id_to_index"])),
        "positive_samples": int(np.sum(labels == 1)),
        "negative_samples": int(np.sum(labels == 0)),
        "config": {
            "c": svm_cfg.c,
            "max_iter": svm_cfg.max_iter,
            "random_state": svm_cfg.random_state,
            "negative_sampling_ratio": svm_cfg.negative_sampling_ratio,
            "use_metadata_features": svm_cfg.use_metadata_features,
            "release_year_bucket_size": svm_cfg.release_year_bucket_size,
        },
    }
    with open(assets_dir / SVM_MODEL_INFO_FILENAME, "w") as handle:
        json.dump(model_info, handle, indent=2)

    print("SVM model training complete and saved.")
