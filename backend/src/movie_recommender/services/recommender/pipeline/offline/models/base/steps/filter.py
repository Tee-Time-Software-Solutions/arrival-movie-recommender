import pandas as pd

from movie_recommender.services.recommender.utils.schema import Config


def iterative_core_filter(
    df: pd.DataFrame, min_user: int, min_movie: int
) -> pd.DataFrame:
    """Remove users/movies below rating thresholds until stable."""
    while True:
        before = len(df)
        valid_movies = df.groupby("movie_id").size()
        df = df[df["movie_id"].isin(valid_movies[valid_movies >= min_movie].index)]
        valid_users = df.groupby("user_id").size()
        df = df[df["user_id"].isin(valid_users[valid_users >= min_user].index)]
        if len(df) == before:
            break
    return df


def run(config: Config) -> None:
    input_path = config.data_dirs.processed_dir / "ratings_clean.parquet"
    output_path = config.data_dirs.processed_dir / "ratings_filtered.parquet"

    df = pd.read_parquet(input_path)
    print(
        f"Before: {len(df)} ratings, {df['user_id'].nunique()} users, {df['movie_id'].nunique()} movies"
    )
    df = iterative_core_filter(
        df, config.pipeline.min_user_ratings, config.pipeline.min_movie_ratings
    )
    print(
        f"After:  {len(df)} ratings, {df['user_id'].nunique()} users, {df['movie_id'].nunique()} movies"
    )
    df.to_parquet(output_path, index=False)
