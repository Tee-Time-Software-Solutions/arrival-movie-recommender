import json
import os

import implicit
import numpy as np
from scipy.sparse import load_npz

from movie_recommender.services.recommender.data_processing.swipe_export import (
    get_app_user_id_offset,
)
from movie_recommender.services.recommender.paths_dev import (
    DATA_PROCESSED,
    ENV_ARTIFACT_VERSION,
    artifacts_dir,
)

PREPROCESS_METADATA_PATH = DATA_PROCESSED / "preprocess_metadata.json"


# Hyperparameters
FACTORS = 64
REGULARIZATION = 0.1
ITERATIONS = 15
ALPHA = 15


def train():
    artifact_root = artifacts_dir()
    matrix_path = artifact_root / "R_train.npz"
    movie_embeddings_path = artifact_root / "movie_embeddings.npy"
    user_embeddings_path = artifact_root / "user_embeddings.npy"
    model_info_path = artifact_root / "model_info.json"

    print("Loading sparse matrix...")
    R_train = load_npz(matrix_path)

    print("Converting to confidence matrix...")

    # Build confidence matrix
    C = R_train.copy()
    C.data = 1 + ALPHA * np.abs(C.data)

    print("Training implicit ALS model...")
    model = implicit.als.AlternatingLeastSquares(
        factors=FACTORS,
        regularization=REGULARIZATION,
        iterations=ITERATIONS,
        use_gpu=False,
    )

    model.fit(C)

    print("Training complete.")

    # Extract embeddings (implicit: item_factors=movies, user_factors=users)
    movie_embeddings = model.item_factors.astype(np.float32)
    user_embeddings = model.user_factors.astype(np.float32)

    print(f"Movie embeddings shape: {movie_embeddings.shape}")
    print(f"User embeddings shape: {user_embeddings.shape}")

    artifact_root.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    np.save(movie_embeddings_path, movie_embeddings)
    np.save(user_embeddings_path, user_embeddings)

    preprocess_meta: dict = {}
    if PREPROCESS_METADATA_PATH.exists():
        with open(PREPROCESS_METADATA_PATH, encoding="utf-8") as f:
            preprocess_meta = json.load(f)

    # Save model metadata
    model_info = {
        "factors": FACTORS,
        "regularization": REGULARIZATION,
        "iterations": ITERATIONS,
        "alpha": ALPHA,
        "num_movies": movie_embeddings.shape[0],
        "num_users": user_embeddings.shape[0],
        "embedding_dim": movie_embeddings.shape[1],
        "artifact_version": artifact_root.name
        if artifact_root.name != "artifacts"
        else "",
        "recommender_artifact_version_env": os.environ.get(ENV_ARTIFACT_VERSION, ""),
        "app_user_id_offset": preprocess_meta.get(
            "app_user_id_offset", get_app_user_id_offset()
        ),
        "interactions_clean_count": preprocess_meta.get("interactions_clean_count"),
        "interactions_clean_fingerprint": preprocess_meta.get(
            "interactions_clean_fingerprint"
        ),
        "skips_for_ranking_count": preprocess_meta.get("skips_for_ranking_count"),
    }

    with open(model_info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4)

    print("Artifacts saved successfully.")


if __name__ == "__main__":
    train()
