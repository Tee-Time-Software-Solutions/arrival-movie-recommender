# recommender/training/build_matrix.py

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from movie_recommender.services.recommender.paths_dev import DATA_SPLITS, ARTIFACTS

TRAIN_PATH = DATA_SPLITS / "train.parquet"
MATRIX_PATH = ARTIFACTS / "R_train.npz"
MAPPINGS_PATH = ARTIFACTS / "mappings.json"


def build_sparse_matrix():
    print("Loading train split...")
    df = pd.read_parquet(TRAIN_PATH)

    print(f"Train interactions: {len(df)}")

    # ---- Create mappings ----
    print("Creating ID mappings...")

    unique_users = df["user_id"].unique()
    unique_movies = df["movie_id"].unique()

    user_id_to_index = {uid: idx for idx, uid in enumerate(unique_users)}
    movie_id_to_index = {mid: idx for idx, mid in enumerate(unique_movies)}
    index_to_movie_id = {idx: mid for mid, idx in movie_id_to_index.items()}

    num_users = len(unique_users)
    num_movies = len(unique_movies)

    print(f"Number of users (train): {num_users}")
    print(f"Number of movies (train): {num_movies}")

    # ---- Map to indices ----
    print("Mapping IDs to indices...")

    user_indices = df["user_id"].map(user_id_to_index).values
    movie_indices = df["movie_id"].map(movie_id_to_index).values
    preferences = df["preference"].values

    # ---- Build sparse matrix ----
    print("Building CSR sparse matrix...")

    R_train = csr_matrix(
        (preferences, (user_indices, movie_indices)),
        shape=(num_users, num_movies),
        dtype=np.float32
    )

    print("Matrix built.")
    print(f"Matrix shape: {R_train.shape}")
    print(f"Non-zero entries: {R_train.nnz}")

    # ---- Save artifacts ----
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Saving sparse matrix...")
    save_npz(MATRIX_PATH, R_train)

    print("Saving mappings...")
    mappings = {
        "user_id_to_index": user_id_to_index,
        "movie_id_to_index": movie_id_to_index,
        "index_to_movie_id": index_to_movie_id,
    }

    with open(MAPPINGS_PATH, "w") as f:
        json.dump(mappings, f)

    print("Artifacts saved successfully.")


if __name__ == "__main__":
    build_sparse_matrix()
