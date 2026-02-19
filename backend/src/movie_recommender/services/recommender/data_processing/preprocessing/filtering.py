# recommender/training/filtering.py

from movie_recommender.services.recommender.paths_dev import DATA_PROCESSED
from pathlib import Path
import pandas as pd

PROCESSED_INPUT = DATA_PROCESSED / "interactions_clean.parquet"
PROCESSED_OUTPUT = DATA_PROCESSED / "interactions_filtered.parquet"

MIN_USER_INTERACTIONS = 10
MIN_MOVIE_INTERACTIONS = 20


def iterative_core_filter(df, min_user, min_movie):
    """
    Iteratively remove users and movies below interaction thresholds
    until stable.
    """
    iteration = 0

    while True:
        iteration += 1
        print(f"\nIteration {iteration}")

        initial_count = len(df)

        # Filter movies
        movie_counts = df.groupby("movie_id").size()
        valid_movies = movie_counts[movie_counts >= min_movie].index
        df = df[df["movie_id"].isin(valid_movies)]

        # Filter users
        user_counts = df.groupby("user_id").size()
        valid_users = user_counts[user_counts >= min_user].index
        df = df[df["user_id"].isin(valid_users)]

        final_count = len(df)

        print(f"Remaining interactions: {final_count}")

        if final_count == initial_count:
            print("Filtering stabilized.")
            break

    return df


def run_filtering():
    print("Loading interactions_clean.parquet...")
    df = pd.read_parquet(PROCESSED_INPUT)

    print(f"Initial interactions: {len(df)}")
    print(f"Initial users: {df['user_id'].nunique()}")
    print(f"Initial movies: {df['movie_id'].nunique()}")

    df_filtered = iterative_core_filter(
        df, MIN_USER_INTERACTIONS, MIN_MOVIE_INTERACTIONS
    )

    print("\nFinal stats:")
    print(f"Interactions: {len(df_filtered)}")
    print(f"Users: {df_filtered['user_id'].nunique()}")
    print(f"Movies: {df_filtered['movie_id'].nunique()}")

    PROCESSED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_parquet(PROCESSED_OUTPUT, index=False)

    print("\nSaved interactions_filtered.parquet")


if __name__ == "__main__":
    run_filtering()
