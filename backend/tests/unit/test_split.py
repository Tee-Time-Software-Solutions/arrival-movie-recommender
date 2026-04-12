import pandas as pd
import pytest

from movie_recommender.services.recommender.pipeline.offline.models.base.steps.split import (
    chronological_split,
    run,
)
from movie_recommender.services.recommender.utils.schema import Config, DataConfig


def _make_config(tmp_path):
    return Config(data_dirs=DataConfig(
        source_dir=tmp_path,
        processed_dir=tmp_path,
        splits_dir=tmp_path,
        model_assets_dir=tmp_path,
    ))


def _make_user_df(user_id, n, start_ts=1000):
    return pd.DataFrame({
        "user_id": [user_id] * n,
        "movie_id": list(range(n)),
        "preference": [1] * n,
        "timestamp": list(range(start_ts, start_ts + n)),
    })


TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


class TestChronologicalSplit:
    def test_respects_approximate_ratio(self):
        df = _make_user_df(1, 100)
        train, val, test = chronological_split(df, TRAIN_RATIO, VAL_RATIO)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_split_is_per_user(self):
        df = pd.concat([_make_user_df(1, 20), _make_user_df(2, 20)])
        train, val, test = chronological_split(df, TRAIN_RATIO, VAL_RATIO)
        assert set(train["user_id"].unique()) == {1, 2}

    def test_ordering_is_chronological(self):
        df = _make_user_df(1, 50, start_ts=100)
        train, val, test = chronological_split(df, TRAIN_RATIO, VAL_RATIO)
        assert train["timestamp"].max() <= val["timestamp"].min()
        assert val["timestamp"].max() <= test["timestamp"].min()

    def test_users_with_few_interactions_raise(self):
        df = pd.concat([_make_user_df(1, 2), _make_user_df(2, 20)])
        with pytest.raises(AssertionError):
            chronological_split(df, TRAIN_RATIO, VAL_RATIO)

    def test_total_rows_preserved(self):
        df = pd.concat([_make_user_df(1, 30), _make_user_df(2, 50)])
        train, val, test = chronological_split(df, TRAIN_RATIO, VAL_RATIO)
        assert len(train) + len(val) + len(test) == 80

    def test_single_user_many_interactions(self):
        df = _make_user_df(1, 200)
        train, val, test = chronological_split(df, TRAIN_RATIO, VAL_RATIO)
        assert len(train) > 0 and len(val) > 0 and len(test) > 0

    def test_multiple_users_varying_counts(self):
        df = pd.concat([_make_user_df(1, 10), _make_user_df(2, 50), _make_user_df(3, 100)])
        train, val, test = chronological_split(df, TRAIN_RATIO, VAL_RATIO)
        assert len(train) + len(val) + len(test) == 160


class TestRunSplitOrchestration:
    def _run(self, tmp_path, df):
        input_path = tmp_path / "ratings_filtered.parquet"
        df.to_parquet(input_path, index=False)
        config = _make_config(tmp_path)
        run(config)
        return tmp_path / "train.parquet", tmp_path / "val.parquet", tmp_path / "test.parquet"

    def test_three_files_created(self, tmp_path):
        df = pd.concat([_make_user_df(1, 20), _make_user_df(2, 20)])
        train_p, val_p, test_p = self._run(tmp_path, df)
        assert train_p.exists() and val_p.exists() and test_p.exists()

    def test_total_rows_preserved(self, tmp_path):
        df = pd.concat([_make_user_df(1, 30), _make_user_df(2, 50)])
        train_p, val_p, test_p = self._run(tmp_path, df)
        total = sum(len(pd.read_parquet(p)) for p in [train_p, val_p, test_p])
        assert total == 80
