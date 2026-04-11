import pandas as pd
import pytest

from movie_recommender.services.recommender.pipeline.models.base.steps.filter import (
    iterative_core_filter,
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


def _make_df(rows):
    return pd.DataFrame(rows, columns=["user_id", "movie_id"])


class TestIterativeCoreFilter:
    def test_all_data_survives_above_thresholds(self):
        rows = [(1, 10), (1, 20), (1, 10), (2, 10), (2, 20), (2, 20)]
        df = _make_df(rows)
        result = iterative_core_filter(df, min_user=2, min_movie=2)
        assert len(result) == len(df)

    def test_sparse_user_removed(self):
        rows = [(1, 10), (1, 20), (1, 10), (2, 10)]
        df = _make_df(rows)
        result = iterative_core_filter(df, min_user=2, min_movie=1)
        assert set(result["user_id"].unique()) == {1}

    def test_sparse_movie_removed(self):
        rows = [(1, 10), (1, 10), (1, 20), (2, 10), (2, 10)]
        df = _make_df(rows)
        result = iterative_core_filter(df, min_user=1, min_movie=2)
        assert set(result["movie_id"].unique()) == {10}

    def test_cascade_removal(self):
        rows = [(1, 10), (1, 10), (2, 20)]
        df = _make_df(rows)
        result = iterative_core_filter(df, min_user=2, min_movie=2)
        assert set(result["user_id"].unique()) == {1}
        assert set(result["movie_id"].unique()) == {10}

    def test_empty_input(self):
        df = _make_df([])
        result = iterative_core_filter(df, min_user=5, min_movie=5)
        assert len(result) == 0

    def test_single_iteration_stabilises(self):
        rows = [(u, m) for u in range(3) for m in range(3)]
        df = _make_df(rows)
        result = iterative_core_filter(df, min_user=3, min_movie=3)
        assert len(result) == 9


class TestRunFilteringOrchestration:
    def _run(self, tmp_path, df):
        input_path = tmp_path / "ratings_clean.parquet"
        df.to_parquet(input_path, index=False)
        config = _make_config(tmp_path)
        run(config)
        return tmp_path / "ratings_filtered.parquet"

    def test_output_created(self, tmp_path):
        rows = [(u, m) for u in range(15) for m in range(25)]
        df = _make_df(rows)
        out = self._run(tmp_path, df)
        assert out.exists()

    def test_sparse_data_filtered(self, tmp_path):
        rows = [(u, m) for u in range(15) for m in range(25)]
        rows.append((99, 0))
        rows.append((0, 999))
        df = _make_df(rows)
        out = self._run(tmp_path, df)
        result = pd.read_parquet(out)
        assert 99 not in result["user_id"].values
        assert 999 not in result["movie_id"].values
