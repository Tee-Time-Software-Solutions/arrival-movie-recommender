from __future__ import annotations

import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz

from movie_recommender.services.recommender.paths_dev import (
    ARTIFACTS,
    DATA_PROCESSED,
    DATA_SPLITS,
)


TRAIN_PATH = DATA_SPLITS / "train.parquet"
VAL_PATH = DATA_SPLITS / "val.parquet"
MOVIES_FILTERED_PATH = DATA_PROCESSED / "movies_filtered.parquet"

INTERACTIONS_PATH = ARTIFACTS / "fm_interactions.npz"
ITEM_FEATURES_PATH = ARTIFACTS / "fm_item_features.npz"
MAPPINGS_PATH = ARTIFACTS / "fm_mappings.json"
ITEM_FEATURE_INDEX_PATH = ARTIFACTS / "fm_item_feature_index.json"


def _extract_genres(genres_value) -> Tuple[str, ...]:
    """
    Extract a tuple of genre strings from the stored value.

    Handles:
        - None / NaN
        - pipe-separated strings like "Drama|Crime"
        - list/array of genres
    """
    # Missing value
    if genres_value is None:
        return ()

    # Scalar NaN
    if isinstance(genres_value, float) and pd.isna(genres_value):
        return ()

    # Already a list/tuple/array of genres
    if isinstance(genres_value, (list, tuple, np.ndarray, pd.Series)):
        cleaned: list[str] = []
        for g in genres_value:
            if g is None:
                continue
            if isinstance(g, float) and pd.isna(g):
                continue
            s = str(g).strip()
            if s:
                cleaned.append(s)
        return tuple(cleaned)

    # Pipe-separated string
    if isinstance(genres_value, str):
        parts = [g.strip() for g in genres_value.split("|") if g.strip()]
        return tuple(parts)

    # Fallback: single value coerced to string
    return (str(genres_value).strip(),) if str(genres_value).strip() else ()


def build_lightfm_data() -> None:
    """
    Build interactions and item feature matrices for LightFM.

    - interactions: CSR [num_users, num_items] with 1.0 for positive interactions
                    (preference > 0 in train split)
    - item_features: CSR [num_items, num_item_features] with binary indicators for
                     genres and release year.
    - mappings: user_id/index and movie_id/index mappings for FM.
    """
    print("Loading train interactions for LightFM...")
    train_df = pd.read_parquet(TRAIN_PATH)

    print("Filtering to positive interactions (preference > 0)...")
    pos_df = train_df[train_df["preference"] > 0].copy()
    print(f"Positive interactions: {len(pos_df)}")

    print("Building user and movie ID mappings...")
    unique_users = pos_df["user_id"].unique()
    unique_movies = pos_df["movie_id"].unique()

    user_id_to_index = {int(uid): idx for idx, uid in enumerate(unique_users)}
    movie_id_to_index = {int(mid): idx for idx, mid in enumerate(unique_movies)}

    num_users = len(user_id_to_index)
    num_items = len(movie_id_to_index)

    print(f"LightFM users: {num_users}, items: {num_items}")

    # Interactions matrix
    row = pos_df["user_id"].map(user_id_to_index).to_numpy()
    col = pos_df["movie_id"].map(movie_id_to_index).to_numpy()
    data = np.ones_like(row, dtype=np.float32)

    interactions = csr_matrix(
        (data, (row, col)),
        shape=(num_users, num_items),
        dtype=np.float32,
    )

    print("Saving interactions matrix...")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    save_npz(INTERACTIONS_PATH, interactions)

    # Item features
    print("Loading filtered movies metadata...")
    movies_df = pd.read_parquet(MOVIES_FILTERED_PATH)
    movies_df = movies_df[movies_df["movie_id"].isin(unique_movies)].copy()

    print("Building item feature index (genres + year)...")
    feature_to_index: Dict[str, int] = {}

    def add_feature(name: str) -> int:
        if name not in feature_to_index:
            feature_to_index[name] = len(feature_to_index)
        return feature_to_index[name]

    # First pass: collect all features
    for _, row_m in movies_df.iterrows():
        mid = int(row_m["movie_id"])
        if mid not in movie_id_to_index:
            continue

        year_val = row_m.get("release_year")
        if not pd.isna(year_val):
            add_feature(f"year_{int(year_val)}")

        genres_val = row_m.get("genres")
        for g in _extract_genres(genres_val):
            add_feature(f"genre_{g}")

    num_item_features = len(feature_to_index)
    print(f"Total item features: {num_item_features}")

    # Second pass: build CSR
    data_if: list[float] = []
    row_if: list[int] = []
    col_if: list[int] = []

    for _, row_m in movies_df.iterrows():
        mid = int(row_m["movie_id"])
        item_idx = movie_id_to_index.get(mid)
        if item_idx is None:
            continue

        year_val = row_m.get("release_year")
        if not pd.isna(year_val):
            f_idx = add_feature(f"year_{int(year_val)}")
            row_if.append(item_idx)
            col_if.append(f_idx)
            data_if.append(1.0)

        genres_val = row_m.get("genres")
        for g in _extract_genres(genres_val):
            f_idx = add_feature(f"genre_{g}")
            row_if.append(item_idx)
            col_if.append(f_idx)
            data_if.append(1.0)

    item_features = csr_matrix(
        (np.array(data_if, dtype=np.float32), (np.array(row_if), np.array(col_if))),
        shape=(num_items, num_item_features),
        dtype=np.float32,
    )

    print("Saving item features matrix and mappings...")
    save_npz(ITEM_FEATURES_PATH, item_features)

    mappings = {
        "user_id_to_index": user_id_to_index,
        "movie_id_to_index": movie_id_to_index,
    }
    with open(MAPPINGS_PATH, "w") as f:
        json.dump(mappings, f)

    with open(ITEM_FEATURE_INDEX_PATH, "w") as f:
        json.dump({"feature_to_index": feature_to_index}, f)

    print("LightFM data artifacts saved.")


def load_lightfm_data():
    interactions = load_npz(INTERACTIONS_PATH)
    item_features = load_npz(ITEM_FEATURES_PATH)
    with open(MAPPINGS_PATH, "r") as f:
        mappings = json.load(f)
    return interactions, item_features, mappings


