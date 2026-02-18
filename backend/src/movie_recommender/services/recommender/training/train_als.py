# recommender/training/train_als.py

from pathlib import Path
import json
import numpy as np
from scipy.sparse import load_npz
import implicit
from movie_recommender.services.recommender.paths_dev import ARTIFACTS


MATRIX_PATH = ARTIFACTS / "R_train.npz"
EMBEDDINGS_PATH = ARTIFACTS / "movie_embeddings.npy"
MODEL_INFO_PATH = ARTIFACTS / "model_info.json"


# Hyperparameters
FACTORS = 64
REGULARIZATION = 0.1
ITERATIONS = 15
ALPHA = 15


def train():
    print("Loading sparse matrix...")
    R_train = load_npz(MATRIX_PATH)

    print("Converting to confidence matrix...")

    # Build confidence matrix
    C = R_train.copy()
    C.data = 1 + ALPHA * np.abs(C.data)

    print("Training implicit ALS model...")
    model = implicit.als.AlternatingLeastSquares(
        factors=FACTORS,
        regularization=REGULARIZATION,
        iterations=ITERATIONS,
        use_gpu=False
    )

    model.fit(C)

    print("Training complete.")

    # Extract movie embeddings
    # model.item_factors corresponds to movies
    movie_embeddings = model.item_factors

    print(f"Movie embeddings shape: {movie_embeddings.shape}")

    # Save embeddings
    np.save(EMBEDDINGS_PATH, movie_embeddings)

    # Save model metadata
    model_info = {
        "factors": FACTORS,
        "regularization": REGULARIZATION,
        "iterations": ITERATIONS,
        "alpha": ALPHA,
        "num_movies": movie_embeddings.shape[0],
        "embedding_dim": movie_embeddings.shape[1]
    }

    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(model_info, f, indent=4)

    print("Artifacts saved successfully.")


if __name__ == "__main__":
    train()
