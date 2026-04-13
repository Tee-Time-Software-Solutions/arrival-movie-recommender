from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz

from movie_recommender.services.recommender.utils.schema import Config

SVM_FEATURES_FILENAME = "svm_train_features.npz"
SVM_LABELS_FILENAME = "svm_train_labels.npy"
SVM_MAPPINGS_FILENAME = "svm_feature_mappings.json"


def _extract_genres(genres_value: Any) -> tuple[str, ...]:
    if genres_value is None:
        return ()
    if isinstance(genres_value, float) and pd.isna(genres_value):
        return ()
    if isinstance(genres_value, (list, tuple, np.ndarray, pd.Series)):
        return tuple(
            str(genre).strip()
            for genre in genres_value
            if genre
            and not (isinstance(genre, float) and pd.isna(genre))
            and str(genre).strip()
        )
    if isinstance(genres_value, str):
        return tuple(
            genre.strip() for genre in genres_value.split("|") if genre.strip()
        )
    normalized = str(genres_value).strip()
    return (normalized,) if normalized else ()


def _year_bucket(release_year: Any, bucket_size: int) -> str | None:
    if release_year is None:
        return None
    if isinstance(release_year, float) and pd.isna(release_year):
        return None
    year = int(release_year)
    start = (year // bucket_size) * bucket_size
    end = start + bucket_size - 1
    return f"{start}_{end}"


def _build_movie_metadata_features(
    movies_df: pd.DataFrame, *, use_metadata_features: bool, year_bucket_size: int
) -> tuple[dict[str, int], dict[str, int], dict[int, list[int]]]:
    if not use_metadata_features:
        return {}, {}, {}

    genre_to_index: dict[str, int] = {}
    year_bucket_to_index: dict[str, int] = {}
    movie_metadata_feature_indices: dict[int, list[int]] = {}

    for _, row in movies_df.iterrows():
        movie_id = int(row["movie_id"])
        local_indices: list[int] = []

        for genre in _extract_genres(row.get("genres")):
            if genre not in genre_to_index:
                genre_to_index[genre] = len(genre_to_index) + len(year_bucket_to_index)
            local_indices.append(genre_to_index[genre])

        bucket = _year_bucket(row.get("release_year"), year_bucket_size)
        if bucket is not None:
            if bucket not in year_bucket_to_index:
                year_bucket_to_index[bucket] = len(genre_to_index) + len(
                    year_bucket_to_index
                )
            local_indices.append(year_bucket_to_index[bucket])

        movie_metadata_feature_indices[movie_id] = sorted(set(local_indices))

    return genre_to_index, year_bucket_to_index, movie_metadata_feature_indices


def _append_feature_row(
    *,
    row_index: int,
    user_id: int,
    movie_id: int,
    mappings: dict[str, Any],
    rows: list[int],
    cols: list[int],
    values: list[float],
) -> None:
    user_idx = mappings["user_id_to_index"][user_id]
    movie_idx = mappings["movie_id_to_index"][movie_id]
    feature_layout = mappings["feature_layout"]

    rows.extend([row_index, row_index])
    cols.extend(
        [
            feature_layout["user_offset"] + user_idx,
            feature_layout["movie_offset"] + movie_idx,
        ]
    )
    values.extend([1.0, 1.0])

    metadata_local_indices = mappings["movie_metadata_feature_indices"].get(
        movie_id, []
    )
    for local_idx in metadata_local_indices:
        rows.append(row_index)
        cols.append(feature_layout["metadata_offset"] + local_idx)
        values.append(1.0)


def run(config: Config) -> None:
    """Prepare sparse SVM training data using positives + sampled negatives."""
    assets_dir = config.data_dirs.model_assets_dir
    train_df = pd.read_parquet(config.data_dirs.splits_dir / "train.parquet")
    movies_df = pd.read_parquet(
        config.data_dirs.processed_dir / "movies_filtered.parquet"
    )

    train_df["user_id"] = train_df["user_id"].astype(int)
    train_df["movie_id"] = train_df["movie_id"].astype(int)
    movies_df["movie_id"] = movies_df["movie_id"].astype(int)

    positive_df = train_df[train_df["preference"] > 0].copy()
    unique_users = sorted(train_df["user_id"].unique())
    unique_movies = sorted(movies_df["movie_id"].unique())
    if not unique_movies:
        unique_movies = sorted(train_df["movie_id"].unique())

    user_id_to_index = {int(user_id): idx for idx, user_id in enumerate(unique_users)}
    movie_id_to_index = {
        int(movie_id): idx for idx, movie_id in enumerate(unique_movies)
    }

    svm_cfg = config.models.svm
    genre_to_index, year_bucket_to_index, movie_metadata_feature_indices = (
        _build_movie_metadata_features(
            movies_df,
            use_metadata_features=svm_cfg.use_metadata_features,
            year_bucket_size=svm_cfg.release_year_bucket_size,
        )
    )
    metadata_feature_count = len(genre_to_index) + len(year_bucket_to_index)

    feature_layout = {
        "user_offset": 0,
        "movie_offset": len(user_id_to_index),
        "metadata_offset": len(user_id_to_index) + len(movie_id_to_index),
        "num_features": len(user_id_to_index)
        + len(movie_id_to_index)
        + metadata_feature_count,
    }
    mappings: dict[str, Any] = {
        "user_id_to_index": user_id_to_index,
        "movie_id_to_index": movie_id_to_index,
        "index_to_movie_id": {
            idx: movie_id for movie_id, idx in movie_id_to_index.items()
        },
        "genre_to_index": genre_to_index,
        "year_bucket_to_index": year_bucket_to_index,
        "movie_metadata_feature_indices": movie_metadata_feature_indices,
        "feature_layout": feature_layout,
        "use_metadata_features": svm_cfg.use_metadata_features,
        "negative_sampling_ratio": svm_cfg.negative_sampling_ratio,
    }

    print(f"SVM positives in train split: {len(positive_df)}")
    print(
        f"SVM feature dimensions: users={len(user_id_to_index)}, "
        f"movies={len(movie_id_to_index)}, metadata={metadata_feature_count}, "
        f"total={feature_layout['num_features']}"
    )

    positives_by_user = (
        positive_df.groupby("user_id")["movie_id"]
        .apply(lambda x: set(int(m) for m in x))
        .to_dict()
    )
    all_movies = np.array(unique_movies, dtype=np.int64)
    rng = np.random.default_rng(svm_cfg.random_state)

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    labels: list[int] = []
    row_idx = 0

    for user_id in unique_users:
        user_positives = positives_by_user.get(user_id, set())
        if not user_positives:
            continue

        for movie_id in user_positives:
            if movie_id not in movie_id_to_index:
                continue
            _append_feature_row(
                row_index=row_idx,
                user_id=user_id,
                movie_id=movie_id,
                mappings=mappings,
                rows=rows,
                cols=cols,
                values=values,
            )
            labels.append(1)
            row_idx += 1

        available_negatives = all_movies[~np.isin(all_movies, list(user_positives))]
        requested_negatives = int(len(user_positives) * svm_cfg.negative_sampling_ratio)
        negative_count = min(len(available_negatives), requested_negatives)
        if negative_count <= 0:
            continue
        sampled_negatives = rng.choice(
            available_negatives, size=negative_count, replace=False
        )

        for movie_id in sampled_negatives.tolist():
            _append_feature_row(
                row_index=row_idx,
                user_id=user_id,
                movie_id=int(movie_id),
                mappings=mappings,
                rows=rows,
                cols=cols,
                values=values,
            )
            labels.append(0)
            row_idx += 1

    if not labels:
        raise ValueError("SVM data preparation produced zero training samples.")
    if len(np.unique(labels)) < 2:
        raise ValueError(
            "SVM training requires both positive and negative labels. "
            "Increase negative_sampling_ratio or check training data."
        )

    features = csr_matrix(
        (np.asarray(values, dtype=np.float32), (np.asarray(rows), np.asarray(cols))),
        shape=(len(labels), feature_layout["num_features"]),
        dtype=np.float32,
    )
    y = np.asarray(labels, dtype=np.int8)

    save_npz(assets_dir / SVM_FEATURES_FILENAME, features)
    np.save(assets_dir / SVM_LABELS_FILENAME, y)
    with open(assets_dir / SVM_MAPPINGS_FILENAME, "w") as handle:
        json.dump(mappings, handle, indent=2)

    print(
        f"SVM data artifacts saved: X={features.shape}, "
        f"labels={len(y)}, positives={int((y == 1).sum())}, negatives={int((y == 0).sum())}"
    )


def _normalize_mappings(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id_to_index": {
            int(key): int(value) for key, value in raw["user_id_to_index"].items()
        },
        "movie_id_to_index": {
            int(key): int(value) for key, value in raw["movie_id_to_index"].items()
        },
        "index_to_movie_id": {
            int(key): int(value) for key, value in raw["index_to_movie_id"].items()
        },
        "genre_to_index": {
            str(key): int(value) for key, value in raw["genre_to_index"].items()
        },
        "year_bucket_to_index": {
            str(key): int(value) for key, value in raw["year_bucket_to_index"].items()
        },
        "movie_metadata_feature_indices": {
            int(key): [int(v) for v in value]
            for key, value in raw["movie_metadata_feature_indices"].items()
        },
        "feature_layout": {
            "user_offset": int(raw["feature_layout"]["user_offset"]),
            "movie_offset": int(raw["feature_layout"]["movie_offset"]),
            "metadata_offset": int(raw["feature_layout"]["metadata_offset"]),
            "num_features": int(raw["feature_layout"]["num_features"]),
        },
        "use_metadata_features": bool(raw["use_metadata_features"]),
        "negative_sampling_ratio": float(raw["negative_sampling_ratio"]),
    }


def load_svm_training_data(
    config: Config,
) -> tuple[csr_matrix, np.ndarray, dict[str, Any]]:
    assets_dir = config.data_dirs.model_assets_dir
    features = load_npz(assets_dir / SVM_FEATURES_FILENAME)
    labels = np.load(assets_dir / SVM_LABELS_FILENAME)
    with open(assets_dir / SVM_MAPPINGS_FILENAME) as handle:
        mappings = _normalize_mappings(json.load(handle))
    return features, labels, mappings


def build_features_for_user_candidates(
    *,
    user_id: int,
    candidate_movie_ids: list[int],
    mappings: dict[str, Any],
) -> tuple[csr_matrix, list[int]]:
    """Build sparse feature rows for a user and candidate movie list."""
    if user_id not in mappings["user_id_to_index"]:
        return csr_matrix((0, mappings["feature_layout"]["num_features"])), []

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    valid_movie_ids: list[int] = []
    row_idx = 0

    for movie_id in candidate_movie_ids:
        if movie_id not in mappings["movie_id_to_index"]:
            continue
        _append_feature_row(
            row_index=row_idx,
            user_id=user_id,
            movie_id=movie_id,
            mappings=mappings,
            rows=rows,
            cols=cols,
            values=values,
        )
        valid_movie_ids.append(movie_id)
        row_idx += 1

    features = csr_matrix(
        (np.asarray(values, dtype=np.float32), (np.asarray(rows), np.asarray(cols))),
        shape=(len(valid_movie_ids), mappings["feature_layout"]["num_features"]),
        dtype=np.float32,
    )
    return features, valid_movie_ids
