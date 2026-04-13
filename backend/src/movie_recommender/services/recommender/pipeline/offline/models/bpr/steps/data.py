from __future__ import annotations

import json
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz

from movie_recommender.services.recommender.utils.schema import Config


def _extract_genres(genres_value) -> Tuple[str, ...]:
    if genres_value is None:
        return ()
    if isinstance(genres_value, float) and pd.isna(genres_value):
        return ()
    if isinstance(genres_value, (list, tuple, np.ndarray, pd.Series)):
        return tuple(
            str(g).strip()
            for g in genres_value
            if g and not (isinstance(g, float) and pd.isna(g)) and str(g).strip()
        )
    if isinstance(genres_value, str):
        return tuple(g.strip() for g in genres_value.split("|") if g.strip())
    s = str(genres_value).strip()
    return (s,) if s else ()


def run(config: Config) -> None:
    """Build interactions + id mappings for implicit BPR."""
    assets_dir = config.data_dirs.model_assets_dir

    train_df = pd.read_parquet(config.data_dirs.splits_dir / "train.parquet")
    pos_df = train_df[train_df["preference"] > 0].copy()
    print(f"Positive interactions: {len(pos_df)}")

    unique_users = pos_df["user_id"].unique()
    unique_movies = pos_df["movie_id"].unique()

    user_id_to_index = {int(uid): idx for idx, uid in enumerate(unique_users)}
    movie_id_to_index = {int(mid): idx for idx, mid in enumerate(unique_movies)}

    num_users, num_items = len(user_id_to_index), len(movie_id_to_index)
    print(f"BPR users: {num_users}, items: {num_items}")

    row = pos_df["user_id"].map(user_id_to_index).to_numpy()
    col = pos_df["movie_id"].map(movie_id_to_index).to_numpy()
    interactions = csr_matrix(
        (np.ones_like(row, dtype=np.float32), (row, col)),
        shape=(num_users, num_items),
        dtype=np.float32,
    )

    save_npz(assets_dir / "bpr_interactions.npz", interactions)

    with open(assets_dir / "bpr_mappings.json", "w") as f:
        json.dump(
            {
                "user_id_to_index": user_id_to_index,
                "movie_id_to_index": movie_id_to_index,
            },
            f,
        )

    print("BPR data artifacts saved.")


def load_bpr_data(config: Config):
    assets_dir = config.data_dirs.model_assets_dir
    interactions = load_npz(assets_dir / "bpr_interactions.npz")
    with open(assets_dir / "bpr_mappings.json") as f:
        mappings = json.load(f)
    return interactions, mappings
