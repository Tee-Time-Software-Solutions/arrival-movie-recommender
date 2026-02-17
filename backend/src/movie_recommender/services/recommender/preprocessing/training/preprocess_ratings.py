"""Transform raw ratings.csv into swipe-style interactions."""

from pathlib import Path

import pandas as pd

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[7]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "ml-20m" / "ratings.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "interactions_clean.parquet"


def main() -> None:
    # 1. Load with correct dtypes
    df = pd.read_csv(
        RAW_PATH,
        dtype={
            "userId": "int32",
            "movieId": "int32",
            "rating": "float32",
            "timestamp": "int64",
        },
    )

    # 2. Rename columns
    df = df.rename(columns={"userId": "user_id", "movieId": "movie_id"})

    # 3. Convert rating (0–5) → swipe_label (1–4)
    def rating_to_swipe(rating: float) -> int:
        if rating < 2.5:
            return 1
        if rating < 3.5:
            return 2
        if rating < 4.5:
            return 3
        return 4

    df["swipe_label"] = df["rating"].apply(rating_to_swipe)

    # 4. Convert 1–4 → preference (-2 to +2)
    PREFERENCE_MAP = {1: -2, 2: -1, 3: 1, 4: 2}
    df["preference"] = df["swipe_label"].map(PREFERENCE_MAP)

    # 5. Drop original rating and swipe_label
    df = df[["user_id", "movie_id", "preference", "timestamp"]]

    # 6. Sanity checks
    print(f"Total interactions: {len(df):,}")
    print(f"Preference distribution:\n{df['preference'].value_counts().sort_index()}")
    print(f"Unique users: {df['user_id'].nunique():,}")
    print(f"Unique movies: {df['movie_id'].nunique():,}")

    # 7. Save output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
