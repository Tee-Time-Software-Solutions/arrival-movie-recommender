import pandas as pd

from movie_recommender.services.recommender.utils.schema import Config


def run(config: Config) -> None:
    """Prune movie metadata to only include movies present in the filtered ratings."""
    ratings = pd.read_parquet(
        config.data_dirs.processed_dir / "ratings_filtered.parquet"
    )
    movies = pd.read_parquet(config.data_dirs.processed_dir / "movies_clean.parquet")
    movies = movies[movies["movie_id"].isin(ratings["movie_id"].unique())]
    movies.to_parquet(
        config.data_dirs.processed_dir / "movies_filtered.parquet", index=False
    )
    print(f"Movies pruned to {len(movies)} entries.")
