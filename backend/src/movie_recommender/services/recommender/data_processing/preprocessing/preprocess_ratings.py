# recommender/training/preprocess_ratings.py

import pandas as pd
from movie_recommender.services.recommender.paths_dev import DATA_PROCESSED

PROCESSED_PATH = DATA_PROCESSED / "interactions_clean.parquet"


def map_rating_to_bucket(rating: float) -> int:
    """
    Convert 0.5–5.0 rating into 1–4 discrete system.
    """
    if rating < 2.5:
        return 1
    elif rating < 3.5:
        return 2
    elif rating < 4.5:
        return 3
    else:
        return 4


def bucket_to_preference(bucket: int) -> int:
    """
    Convert 1–4 bucket into symmetric preference scale.
    1 -> -2
    2 -> -1
    3 -> +1
    4 -> +2
    """
    return {1: -2, 2: -1, 3: 1, 4: 2}[bucket]


def preprocess_ratings():
    """
    Build interactions_clean.parquet from MovieLens + optional swipes_from_db.parquet.

    Delegates to unified_interactions (MovieLens-only if export file is absent).
    """
    from movie_recommender.services.recommender.data_processing.unified_interactions import (
        run_unified_preprocess,
    )

    print("Preprocessing ratings (unified MovieLens + optional app swipes)...")
    meta = run_unified_preprocess()
    print("Preference distribution (interactions_clean):")
    df = pd.read_parquet(PROCESSED_PATH)
    print(df["preference"].value_counts().sort_index())
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Unique movies: {df['movie_id'].nunique()}")
    return meta


if __name__ == "__main__":
    preprocess_ratings()
