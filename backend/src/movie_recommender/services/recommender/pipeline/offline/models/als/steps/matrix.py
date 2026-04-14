import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

from movie_recommender.services.recommender.utils.schema import Config


def run(config: Config) -> None:
    train_path = config.data_dirs.splits_dir / "train.parquet"
    assets_dir = config.data_dirs.model_assets_dir

    print("Loading train split...")
    df = pd.read_parquet(train_path)

    unique_users = df["user_id"].unique()
    unique_movies = df["movie_id"].unique()

    user_id_to_index = {uid: idx for idx, uid in enumerate(unique_users)}
    movie_id_to_index = {mid: idx for idx, mid in enumerate(unique_movies)}
    index_to_movie_id = {idx: mid for mid, idx in movie_id_to_index.items()}

    user_indices = df["user_id"].map(user_id_to_index).values
    movie_indices = df["movie_id"].map(movie_id_to_index).values

    R_train = csr_matrix(
        (df["preference"].values, (user_indices, movie_indices)),
        shape=(len(unique_users), len(unique_movies)),
        dtype=np.float32,
    )

    print(f"Matrix shape: {R_train.shape}, non-zeros: {R_train.nnz}")

    def _to_native(d: dict) -> dict:
        return {int(k): int(v) for k, v in d.items()}

    save_npz(assets_dir / "R_train.npz", R_train)

    with open(assets_dir / "mappings.json", "w") as f:
        json.dump(
            {
                "user_id_to_index": _to_native(user_id_to_index),
                "movie_id_to_index": _to_native(movie_id_to_index),
                "index_to_movie_id": _to_native(index_to_movie_id),
            },
            f,
        )

    print("Matrix and mappings saved.")
