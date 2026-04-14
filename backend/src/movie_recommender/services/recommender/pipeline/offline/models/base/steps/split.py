import pandas as pd

from movie_recommender.services.recommender.utils.schema import Config


def chronological_split(
    df: pd.DataFrame, train_ratio: float, val_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user chronological split. No temporal leakage."""
    train_list, val_list, test_list = [], [], []
    df = df.sort_values(["user_id", "timestamp"])

    for _, user_df in df.groupby("user_id"):
        n = len(user_df)
        assert n >= 3, (
            f"User with {n} ratings found. You need at least 3 ratings per user. Make sure you have run the filter step"
        )

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        train_list.append(user_df.iloc[:train_end])
        val_list.append(user_df.iloc[train_end:val_end])
        test_list.append(user_df.iloc[val_end:])

    return pd.concat(train_list), pd.concat(val_list), pd.concat(test_list)


def run(config: Config) -> None:
    splits_dir = config.data_dirs.splits_dir
    df = pd.read_parquet(config.data_dirs.processed_dir / "ratings_filtered.parquet")
    train, val, test = chronological_split(
        df, config.pipeline.train_ratio, config.pipeline.val_ratio
    )
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    train.to_parquet(splits_dir / "train.parquet", index=False)
    val.to_parquet(splits_dir / "val.parquet", index=False)
    test.to_parquet(splits_dir / "test.parquet", index=False)
