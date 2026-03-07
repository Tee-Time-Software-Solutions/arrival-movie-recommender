# prune_movies.py (or inside filtering.py)
import pandas as pd
from pathlib import Path
from movie_recommender.services.recommender.paths_dev import DATA_PROCESSED

MOVIES_INPUT = DATA_PROCESSED / "movies_clean.parquet"
MOVIES_OUTPUT = DATA_PROCESSED / "movies_filtered.parquet"
INTERACTIONS_INPUT = DATA_PROCESSED / "interactions_filtered.parquet"


def prune_movies():
    interactions = pd.read_parquet(INTERACTIONS_INPUT)
    movies = pd.read_parquet(MOVIES_INPUT)

    valid_movie_ids = interactions["movie_id"].unique()
    movies = movies[movies["movie_id"].isin(valid_movie_ids)]

    movies.to_parquet(MOVIES_OUTPUT, index=False)

    print(f"Remaining movies: {len(movies)}")


if __name__ == "__main__":
    prune_movies()
