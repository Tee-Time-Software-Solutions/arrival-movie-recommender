from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

from movie_recommender.services.recommender.utils.schema import Config


def _to_native_int_dict(values: dict[int, int]) -> dict[int, int]:
    return {int(key): int(value) for key, value in values.items()}


def run(config: Config) -> None:
    train_path = config.data_dirs.splits_dir / "train.parquet"
    assets_dir = config.data_dirs.model_assets_dir

    print("Loading train split for Item-CF...")
    df = pd.read_parquet(train_path)

    required_columns = {"user_id", "movie_id", "preference"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in train split: {missing}")

    unique_users = df["user_id"].unique()
    unique_movies = df["movie_id"].unique()

    user_id_to_index = {int(uid): idx for idx, uid in enumerate(unique_users)}
    movie_id_to_index = {int(mid): idx for idx, mid in enumerate(unique_movies)}
    index_to_user_id = {idx: uid for uid, idx in user_id_to_index.items()}
    index_to_movie_id = {idx: mid for mid, idx in movie_id_to_index.items()}

    user_indices = df["user_id"].map(user_id_to_index).to_numpy()
    movie_indices = df["movie_id"].map(movie_id_to_index).to_numpy()

    interaction_matrix = csr_matrix(
        (df["preference"].to_numpy(dtype=np.float32), (user_indices, movie_indices)),
        shape=(len(unique_users), len(unique_movies)),
        dtype=np.float32,
    )

    print(
        "Item-CF train matrix shape: "
        f"{interaction_matrix.shape}, non-zeros: {interaction_matrix.nnz}"
    )

    save_npz(assets_dir / "item_cf_train_matrix.npz", interaction_matrix)
    with open(assets_dir / "item_cf_mappings.json", "w") as file_obj:
        json.dump(
            {
                "user_id_to_index": _to_native_int_dict(user_id_to_index),
                "movie_id_to_index": _to_native_int_dict(movie_id_to_index),
                "index_to_user_id": _to_native_int_dict(index_to_user_id),
                "index_to_movie_id": _to_native_int_dict(index_to_movie_id),
            },
            file_obj,
            indent=2,
        )

    print("Item-CF matrix and mappings saved.")
