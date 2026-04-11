"""
Merge MovieLens ratings with exported app swipes into interactions_clean.parquet.

- Signed preference scale -2 .. +2 for both sources (skips excluded from training rows).
- Dedupe (user_id, movie_id) keeping the latest timestamp.
- Writes skips_for_ranking.parquet for future re-ranking (not used by ALS).
- Writes preprocess_metadata.json for model_info enrichment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from movie_recommender.services.recommender.data_processing.preprocessing.preprocess_ratings import (
    bucket_to_preference,
    map_rating_to_bucket,
)
from movie_recommender.services.recommender.data_processing.swipe_export import (
    SWIPES_FROM_DB_FILENAME,
    get_app_user_id_offset,
)
from movie_recommender.services.recommender.paths_dev import DATA_PROCESSED, DATA_RAW

RATINGS_CSV = DATA_RAW / "ratings.csv"
INTERACTIONS_CLEAN_PATH = DATA_PROCESSED / "interactions_clean.parquet"
SKIPS_FOR_RANKING_PATH = DATA_PROCESSED / "skips_for_ranking.parquet"
PREPROCESS_METADATA_PATH = DATA_PROCESSED / "preprocess_metadata.json"


def load_movielens_ratings_df() -> pd.DataFrame:
    df = pd.read_csv(
        RATINGS_CSV,
        dtype={
            "userId": "int32",
            "movieId": "int32",
            "rating": "float32",
            "timestamp": "int64",
        },
    )
    df = df.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    df["bucket"] = df["rating"].apply(map_rating_to_bucket).astype("int8")
    df["preference"] = df["bucket"].apply(bucket_to_preference).astype("int8")
    return df[["user_id", "movie_id", "preference", "timestamp"]]


def load_swipes_export_df(path: Path | None = None) -> pd.DataFrame | None:
    p = path if path is not None else DATA_RAW / SWIPES_FROM_DB_FILENAME
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    required = {"user_id", "movie_id", "preference", "timestamp"}
    if not required.issubset(df.columns):
        raise ValueError(f"Swipes export missing columns {required}, got {df.columns.tolist()}")
    return df


def merge_and_dedupe_interactions(
    movielens_df: pd.DataFrame,
    swipes_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Concatenate sources, exclude skips (preference == 0) from training frame,
    sort by timestamp and keep last per (user_id, movie_id).

    Returns (interactions_for_als, skips_df for ranking sidecar).
    """
    skips = pd.DataFrame(columns=["user_id", "movie_id", "timestamp"])
    parts: list[pd.DataFrame] = [movielens_df.copy()]

    if swipes_df is not None and len(swipes_df) > 0:
        swipe_skips = swipes_df[swipes_df["preference"] == 0][
            ["user_id", "movie_id", "timestamp"]
        ].copy()
        skips = pd.concat([skips, swipe_skips], ignore_index=True)

        app_train = swipes_df[swipes_df["preference"] != 0][
            ["user_id", "movie_id", "preference", "timestamp"]
        ].copy()
        parts.append(app_train)

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values(["user_id", "movie_id", "timestamp"])
    deduped = combined.drop_duplicates(subset=["user_id", "movie_id"], keep="last")
    deduped = deduped.sort_values("timestamp").reset_index(drop=True)

    if len(skips) > 0:
        skips = skips.sort_values(["user_id", "movie_id", "timestamp"])
        skips = skips.drop_duplicates(subset=["user_id", "movie_id"], keep="last")
        skips = skips.reset_index(drop=True)

    return deduped, skips


def _fingerprint_df(df: pd.DataFrame) -> str:
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _print_interaction_snapshot(title: str, df: pd.DataFrame, *, note: str | None = None) -> None:
    """Human-readable stats for one interaction table (MovieLens, swipes export, or merged)."""
    print()
    print(f"--- {title} ---")
    if note:
        print(f"    ({note})")
    print(f"    rows: {len(df):,}")
    if df.empty:
        print("    (empty)")
        return
    print(f"    columns: {list(df.columns)}")
    if "user_id" in df.columns:
        print(f"    unique users:  {df['user_id'].nunique():,}")
    if "movie_id" in df.columns:
        print(f"    unique movies: {df['movie_id'].nunique():,}")
    if "preference" in df.columns:
        vc = df["preference"].value_counts().sort_index()
        parts = [f"{int(k)} → {v:,}" for k, v in vc.items()]
        print(f"    preference counts:  {', '.join(parts)}")
        if (df["preference"] == 0).any():
            n0 = int((df["preference"] == 0).sum())
            print(f"    rows with preference==0 (skips): {n0:,}")
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        print(f"    timestamp (unix s): min={int(ts.min())}  max={int(ts.max())}")


def run_unified_preprocess(
    swipes_parquet_path: Path | None = None,
) -> dict[str, Any]:
    """
    Build interactions_clean.parquet from MovieLens + optional swipes export.
    If swipes file is missing, MovieLens-only (same as legacy preprocess_ratings output shape).
    """
    print("Loading MovieLens ratings...")
    ml_df = load_movielens_ratings_df()
    _print_interaction_snapshot(
        "MovieLens (ratings.csv → preference scale)",
        ml_df,
        note="source for offline training; no skips",
    )

    swipes_df = load_swipes_export_df(swipes_parquet_path)

    if swipes_df is not None:
        _print_interaction_snapshot(
            "App swipes (swipes_from_db.parquet)",
            swipes_df,
            note="includes skips as preference==0; non-zero merged below",
        )
        print(f"\nMerging app swipes: {len(swipes_df):,} rows from export")
    else:
        print()
        print("--- App swipes (swipes_from_db.parquet) ---")
        print("    (not found — unified table is MovieLens-only after dedupe)")

    train_df, skips_df = merge_and_dedupe_interactions(ml_df, swipes_df)

    _print_interaction_snapshot(
        "Unified / ALS training (MovieLens + app, deduped by latest timestamp per user–movie)",
        train_df,
        note="skips excluded; interactions_clean.parquet",
    )
    if len(skips_df) > 0:
        print()
        print("--- Skips sidecar (for ranking / not in ALS matrix) ---")
        print(f"    rows: {len(skips_df):,}")
        if "user_id" in skips_df.columns:
            print(f"    unique users:  {skips_df['user_id'].nunique():,}")
        if "movie_id" in skips_df.columns:
            print(f"    unique movies: {skips_df['movie_id'].nunique():,}")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(INTERACTIONS_CLEAN_PATH, index=False)
    skips_df.to_parquet(SKIPS_FOR_RANKING_PATH, index=False)

    offset = get_app_user_id_offset()
    meta = {
        "app_user_id_offset": offset,
        "movielens_interaction_count": int(len(ml_df)),
        "swipes_export_count": int(len(swipes_df)) if swipes_df is not None else 0,
        "interactions_clean_count": int(len(train_df)),
        "skips_for_ranking_count": int(len(skips_df)),
        "interactions_clean_fingerprint": _fingerprint_df(train_df),
    }
    with open(PREPROCESS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Unified preprocessing complete.")
    print(f"  interactions_clean: {len(train_df)} rows -> {INTERACTIONS_CLEAN_PATH}")
    print(f"  skips_for_ranking: {len(skips_df)} rows -> {SKIPS_FOR_RANKING_PATH}")
    print(f"  metadata -> {PREPROCESS_METADATA_PATH}")
    return meta
