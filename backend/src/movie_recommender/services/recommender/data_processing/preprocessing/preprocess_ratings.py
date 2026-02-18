# recommender/training/preprocess_ratings.py

from pathlib import Path
import pandas as pd
from movie_recommender.services.recommender.paths_dev import DATA_RAW, DATA_PROCESSED

RAW_PATH = DATA_RAW / "ratings.csv"
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
    print("Loading ratings.csv (this may take ~15–25 seconds)...")

    df = pd.read_csv(
        RAW_PATH,
        dtype={
            "userId": "int32",
            "movieId": "int32",
            "rating": "float32",
            "timestamp": "int64",
        },
    )

    # Rename columns
    df = df.rename(columns={
        "userId": "user_id",
        "movieId": "movie_id",
    })

    print("Mapping ratings to swipe buckets...")

    # Convert ratings → 1–4 bucket
    df["bucket"] = df["rating"].apply(map_rating_to_bucket).astype("int8")

    # Convert bucket → preference scale
    df["preference"] = df["bucket"].apply(bucket_to_preference).astype("int8")

    # Keep only necessary columns
    df = df[["user_id", "movie_id", "preference", "timestamp"]]

    # Ensure output directory exists
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Saving processed interactions...")

    df.to_parquet(PROCESSED_PATH, index=False)

    # ---- Sanity Checks ----
    print("Ratings preprocessing complete.")
    print(f"Total interactions: {len(df)}")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Unique movies: {df['movie_id'].nunique()}")
    print("Preference distribution:")
    print(df["preference"].value_counts().sort_index())


if __name__ == "__main__":
    preprocess_ratings()
