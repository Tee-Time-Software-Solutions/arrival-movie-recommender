from __future__ import annotations

import json
from typing import Dict, Tuple

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
    """Build interaction and item feature matrices for LightFM."""
    assets_dir = config.data_dirs.model_assets_dir

    train_df = pd.read_parquet(config.data_dirs.splits_dir / "train.parquet")
    pos_df = train_df[train_df["preference"] > 0].copy()
    print(f"Positive interactions: {len(pos_df)}")

    unique_users = pos_df["user_id"].unique()
    unique_movies = pos_df["movie_id"].unique()

    user_id_to_index = {int(uid): idx for idx, uid in enumerate(unique_users)}
    movie_id_to_index = {int(mid): idx for idx, mid in enumerate(unique_movies)}

    num_users, num_items = len(user_id_to_index), len(movie_id_to_index)
    print(f"LightFM users: {num_users}, items: {num_items}")

    row = pos_df["user_id"].map(user_id_to_index).to_numpy()
    col = pos_df["movie_id"].map(movie_id_to_index).to_numpy()
    interactions = csr_matrix(
        (np.ones_like(row, dtype=np.float32), (row, col)),
        shape=(num_users, num_items),
        dtype=np.float32,
    )

    save_npz(assets_dir / "fm_interactions.npz", interactions)

    movies_df = pd.read_parquet(
        config.data_dirs.processed_dir / "movies_filtered.parquet"
    )
    movies_df = movies_df[movies_df["movie_id"].isin(unique_movies)].copy()

    feature_to_index: Dict[str, int] = {}

    def add_feature(name: str) -> None:
        if name not in feature_to_index:
            feature_to_index[name] = len(feature_to_index)

    for _, row_m in movies_df.iterrows():
        mid = int(row_m["movie_id"])
        if mid not in movie_id_to_index:
            continue
        year_val = row_m.get("release_year")
        if year_val and not pd.isna(year_val):
            add_feature(f"year_{int(year_val)}")
        for g in _extract_genres(row_m.get("genres")):
            add_feature(f"genre_{g}")

    print(f"Total item features: {len(feature_to_index)}")

    data_if, row_if, col_if = [], [], []
    for _, row_m in movies_df.iterrows():
        mid = int(row_m["movie_id"])
        item_idx = movie_id_to_index.get(mid)
        if item_idx is None:
            continue
        year_val = row_m.get("release_year")
        if year_val and not pd.isna(year_val):
            f_idx = feature_to_index.get(f"year_{int(year_val)}")
            if f_idx is not None:
                row_if.append(item_idx)
                col_if.append(f_idx)
                data_if.append(1.0)
        for g in _extract_genres(row_m.get("genres")):
            f_idx = feature_to_index.get(f"genre_{g}")
            if f_idx is not None:
                row_if.append(item_idx)
                col_if.append(f_idx)
                data_if.append(1.0)

    item_features = csr_matrix(
        (np.array(data_if, dtype=np.float32), (np.array(row_if), np.array(col_if))),
        shape=(num_items, len(feature_to_index)),
        dtype=np.float32,
    )

    save_npz(assets_dir / "fm_item_features.npz", item_features)

    with open(assets_dir / "fm_mappings.json", "w") as f:
        json.dump(
            {
                "user_id_to_index": user_id_to_index,
                "movie_id_to_index": movie_id_to_index,
            },
            f,
        )
    with open(assets_dir / "fm_item_feature_index.json", "w") as f:
        json.dump({"feature_to_index": feature_to_index}, f)

    print("LightFM data artifacts saved.")


def load_lightfm_data(config: Config):
    assets_dir = config.data_dirs.model_assets_dir
    interactions = load_npz(assets_dir / "fm_interactions.npz")
    item_features = load_npz(assets_dir / "fm_item_features.npz")
    with open(assets_dir / "fm_mappings.json") as f:
        mappings = json.load(f)
    return interactions, item_features, mappings
