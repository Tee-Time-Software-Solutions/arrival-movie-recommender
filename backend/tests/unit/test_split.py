import pandas as pd
import pytest

from unittest.mock import patch

from movie_recommender.services.recommender.data_processing.split import (
    chronological_split,
    run_split,
)

_MODULE = "movie_recommender.services.recommender.data_processing.split"


def _make_user_df(user_id, n, start_ts=1000):
    return pd.DataFrame({
        "user_id": [user_id] * n,
        "movie_id": list(range(n)),
        "preference": [1] * n,
        "timestamp": list(range(start_ts, start_ts + n)),
    })


class TestChronologicalSplit:
    def test_respects_approximate_ratio(self):
        df = _make_user_df(1, 100)
        train, val, test = chronological_split(df)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_split_is_per_user(self):
        df = pd.concat([_make_user_df(1, 20), _make_user_df(2, 20)])
        train, val, test = chronological_split(df)
        # Each user should have data in train
        assert set(train["user_id"].unique()) == {1, 2}

    def test_ordering_is_chronological(self):
        df = _make_user_df(1, 50, start_ts=100)
        train, val, test = chronological_split(df)
        assert train["timestamp"].max() <= val["timestamp"].min()
        assert val["timestamp"].max() <= test["timestamp"].min()

    def test_users_with_few_interactions_go_to_train(self):
        # User 1 has only 2 interactions (<3) → all go to train
        # User 2 has enough to split normally
        df = pd.concat([_make_user_df(1, 2), _make_user_df(2, 20)])
        train, val, test = chronological_split(df)
        user1_train = train[train["user_id"] == 1]
        user1_val = val[val["user_id"] == 1]
        user1_test = test[test["user_id"] == 1]
        assert len(user1_train) == 2
        assert len(user1_val) == 0
        assert len(user1_test) == 0

    def test_total_rows_preserved(self):
        df = pd.concat([_make_user_df(1, 30), _make_user_df(2, 50)])
        train, val, test = chronological_split(df)
        assert len(train) + len(val) + len(test) == 80

    def test_single_user_many_interactions(self):
        df = _make_user_df(1, 200)
        train, val, test = chronological_split(df)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_multiple_users_varying_counts(self):
        df = pd.concat([
            _make_user_df(1, 10),
            _make_user_df(2, 50),
            _make_user_df(3, 100),
        ])
        train, val, test = chronological_split(df)
        assert len(train) + len(val) + len(test) == 160


class TestRunSplitOrchestration:
    """Integration-style tests for the run_split() orchestration wrapper."""

    def _run(self, tmp_path, df):
        input_path = tmp_path / "input.parquet"
        df.to_parquet(input_path, index=False)
        train_path = tmp_path / "train.parquet"
        val_path = tmp_path / "val.parquet"
        test_path = tmp_path / "test.parquet"

        with patch(f"{_MODULE}.INPUT_PATH", input_path), \
             patch(f"{_MODULE}.TRAIN_PATH", train_path), \
             patch(f"{_MODULE}.VAL_PATH", val_path), \
             patch(f"{_MODULE}.TEST_PATH", test_path):
            run_split()
        return train_path, val_path, test_path

    def test_three_files_created(self, tmp_path):
        df = pd.concat([_make_user_df(1, 20), _make_user_df(2, 20)])
        train_p, val_p, test_p = self._run(tmp_path, df)
        assert train_p.exists()
        assert val_p.exists()
        assert test_p.exists()

    def test_total_rows_preserved(self, tmp_path):
        df = pd.concat([_make_user_df(1, 30), _make_user_df(2, 50)])
        train_p, val_p, test_p = self._run(tmp_path, df)
        total = sum(len(pd.read_parquet(p)) for p in [train_p, val_p, test_p])
        assert total == 80
