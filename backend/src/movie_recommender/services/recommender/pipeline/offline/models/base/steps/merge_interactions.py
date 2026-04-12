"""Merge MovieLens ratings with ``raw/swipes_from_db.parquet`` into one training file.

``swipes_from_db.parquet`` is produced by ``fetch_app_swipes`` (Postgres).  If that
file is missing — e.g. ``SKIP_DB_SWIPE_EXPORT=1`` or fetch failed — this step uses
MovieLens-only ratings.

``drop_duplicates`` keeps one row per (user_id, movie_id): the latest timestamp.

Output: ``processed/ratings_unified.parquet`` (user_id, movie_id, preference, timestamp).
Skips (preference == 0) are excluded from the training frame.
"""

import pandas as pd

from movie_recommender.services.recommender.utils.schema import Config

_SWIPES_FILENAME = "swipes_from_db.parquet"


def merge_and_dedupe_interactions(
    ml_df: pd.DataFrame,
    swipes_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge two interaction DataFrames and return (train, skips).

    Both frames must have columns: user_id, movie_id, preference, timestamp.

    Returns:
        train  — rows with preference != 0  (fed to ALS / FM)
        skips  — rows with preference == 0  (excluded from training)
    """
    combined = pd.concat([ml_df, swipes_df], ignore_index=True)

    combined = (
        combined.sort_values("timestamp", ascending=False)
        .drop_duplicates(subset=["user_id", "movie_id"], keep="first")
        .reset_index(drop=True)
    )

    skips = combined[combined["preference"] == 0].copy()
    train = combined[combined["preference"] != 0].copy()

    return train, skips


def run(config: Config) -> None:
    ratings_path = config.data_dirs.processed_dir / "ratings_clean.parquet"
    swipes_path = config.data_dirs.source_dir.parent / "raw" / _SWIPES_FILENAME
    out_path = config.data_dirs.processed_dir / "ratings_unified.parquet"

    if not ratings_path.exists():
        raise FileNotFoundError(
            f"ratings_clean.parquet not found at {ratings_path}. "
            "Run preprocess_ratings first."
        )

    ml_df = pd.read_parquet(ratings_path)

    if not swipes_path.exists():
        print(f"  No {_SWIPES_FILENAME} found — using MovieLens-only ratings.")
        ml_df.to_parquet(out_path, index=False)
        print(f"  Wrote {len(ml_df):,} rows to {out_path.name}")
        return

    swipes_df = pd.read_parquet(swipes_path)
    print(f"  MovieLens rows : {len(ml_df):,}")
    print(f"  App swipe rows : {len(swipes_df):,}")

    train, skips = merge_and_dedupe_interactions(ml_df, swipes_df)

    train.to_parquet(out_path, index=False)
    print(
        f"  Unified rows   : {len(train):,}  "
        f"(users: {train['user_id'].nunique():,}, "
        f"movies: {train['movie_id'].nunique():,})"
    )
    if len(skips):
        print(
            f"  Skips excluded : {len(skips):,}  (preference == 0, not used for training)"
        )
    print(f"  Wrote {out_path.name}")
