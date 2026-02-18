# recommender/training/split.py

from pathlib import Path
import pandas as pd
from movie_recommender.services.recommender.paths_dev import DATA_PROCESSED, DATA_SPLITS

INPUT_PATH = DATA_PROCESSED / "interactions_filtered.parquet"
TRAIN_PATH = DATA_SPLITS / "train.parquet"
VAL_PATH = DATA_SPLITS / "val.parquet"
TEST_PATH = DATA_SPLITS / "test.parquet"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# Remaining 0.1 automatically test

def chronological_split(df):
    """
    Perform per-user chronological split.
    """
    train_list = []
    val_list = []
    test_list = []

    # Sort once globally for efficiency
    df = df.sort_values(["user_id", "timestamp"])

    for user_id, user_df in df.groupby("user_id"):
        n = len(user_df)

        if n < 3:
            # If extremely small user (unlikely after filtering)
            train_list.append(user_df)
            continue

        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)

        train_list.append(user_df.iloc[:train_end])
        val_list.append(user_df.iloc[train_end:val_end])
        test_list.append(user_df.iloc[val_end:])

    train = pd.concat(train_list)
    val = pd.concat(val_list)
    test = pd.concat(test_list)

    return train, val, test


def run_split():
    print("Loading filtered interactions...")
    df = pd.read_parquet(INPUT_PATH)

    print(f"Total interactions: {len(df)}")
    print("Splitting chronologically per user...")

    train, val, test = chronological_split(df)

    print("\nSplit sizes:")
    print(f"Train: {len(train)}")
    print(f"Validation: {len(val)}")
    print(f"Test: {len(test)}")

    # Save
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(TRAIN_PATH, index=False)
    val.to_parquet(VAL_PATH, index=False)
    test.to_parquet(TEST_PATH, index=False)

    print("\nSaved train/val/test parquet files.")


if __name__ == "__main__":
    run_split()
