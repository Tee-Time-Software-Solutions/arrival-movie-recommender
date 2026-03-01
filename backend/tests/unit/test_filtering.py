import pandas as pd
import pytest

from unittest.mock import patch

from movie_recommender.services.recommender.data_processing.preprocessing.filtering import (
    iterative_core_filter,
    run_filtering,
)

_MODULE = "movie_recommender.services.recommender.data_processing.preprocessing.filtering"


def _make_df(rows):
    return pd.DataFrame(rows, columns=["user_id", "movie_id"])


class TestIterativeCoreFilter:
    def test_all_data_survives_above_thresholds(self):
        # 2 users, 2 movies, each with 3 interactions → above min thresholds of 2
        rows = [
            (1, 10), (1, 20), (1, 10),
            (2, 10), (2, 20), (2, 20),
        ]
        df = _make_df(rows)
        result = iterative_core_filter(df, min_user=2, min_movie=2)
        assert len(result) == len(df)

    def test_sparse_user_removed(self):
        # user 2 has only 1 interaction → removed
        rows = [
            (1, 10), (1, 20), (1, 10),
            (2, 10),
        ]
        df = _make_df(rows)
        result = iterative_core_filter(df, min_user=2, min_movie=1)
        assert set(result["user_id"].unique()) == {1}

    def test_sparse_movie_removed(self):
        # movie 20 has only 1 interaction → removed
        rows = [
            (1, 10), (1, 10), (1, 20),
            (2, 10), (2, 10),
        ]
        df = _make_df(rows)
        result = iterative_core_filter(df, min_user=1, min_movie=2)
        assert set(result["movie_id"].unique()) == {10}

    def test_cascade_removal(self):
        # movie 20 has 1 interaction → removed → user 2 drops to 0 → removed
        rows = [
            (1, 10), (1, 10),
            (2, 20),
        ]
        df = _make_df(rows)
        result = iterative_core_filter(df, min_user=2, min_movie=2)
        assert set(result["user_id"].unique()) == {1}
        assert set(result["movie_id"].unique()) == {10}

    def test_empty_input(self):
        df = _make_df([])
        result = iterative_core_filter(df, min_user=5, min_movie=5)
        assert len(result) == 0

    def test_single_iteration_stabilises(self):
        # Already stable: all users and movies meet thresholds
        rows = [(u, m) for u in range(3) for m in range(3)]
        df = _make_df(rows)
        result = iterative_core_filter(df, min_user=3, min_movie=3)
        assert len(result) == 9


class TestRunFilteringOrchestration:
    """Integration-style tests for the run_filtering() orchestration wrapper."""

    def _run(self, tmp_path, df):
        input_path = tmp_path / "interactions_clean.parquet"
        df.to_parquet(input_path, index=False)
        output_path = tmp_path / "interactions_filtered.parquet"

        with patch(f"{_MODULE}.PROCESSED_INPUT", input_path), \
             patch(f"{_MODULE}.PROCESSED_OUTPUT", output_path):
            run_filtering()
        return output_path

    def test_output_created(self, tmp_path):
        # 15 users x 25 movies → all above default thresholds (10/20)
        rows = [(u, m) for u in range(15) for m in range(25)]
        df = _make_df(rows)
        out = self._run(tmp_path, df)
        assert out.exists()

    def test_sparse_data_filtered(self, tmp_path):
        # Dense core: 15 users x 25 movies (each above MIN_USER=10 and MIN_MOVIE=20)
        rows = [(u, m) for u in range(15) for m in range(25)]
        # Sparse user 99 has only 1 interaction
        rows.append((99, 0))
        # Sparse movie 999 has only 1 interaction
        rows.append((0, 999))
        df = _make_df(rows)
        out = self._run(tmp_path, df)
        result = pd.read_parquet(out)
        assert 99 not in result["user_id"].values
        assert 999 not in result["movie_id"].values
