import pandas as pd

from movie_recommender.services.recommender.utils.schema import Config


def run(config: Config) -> None:
    source_path = config.data_dirs.source_dir / "ratings.csv"
    processed_path = config.data_dirs.processed_dir / "ratings_clean.parquet"

    print("Loading ratings.csv (may take ~20s)...")
    df = pd.read_csv(
        source_path,
        dtype={
            "userId": "int32",
            "movieId": "int32",
            "rating": "float32",
            "timestamp": "int64",
        },
    )
    df = df.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    df["preference"] = pd.cut(
        df["rating"], bins=[0, 2.5, 3.5, 4.5, 5], labels=[-2, -1, 1, 2]
    ).astype("int8")
    df = df[["user_id", "movie_id", "preference", "timestamp"]]

    df.to_parquet(processed_path, index=False)
    print(f"Ratings done. ratings: {len(df)}, users: {df['user_id'].nunique()}")
