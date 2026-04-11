import json
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import load_npz

from movie_recommender.services.recommender.pipeline.models.als.steps.matrix import run
from movie_recommender.services.recommender.utils.schema import Config, DataConfig


def _make_config(tmp_path):
    return Config(data_dirs=DataConfig(
        source_dir=tmp_path,
        processed_dir=tmp_path,
        splits_dir=tmp_path,
        model_assets_dir=tmp_path,
    ))


def _make_train_df(rows):
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "preference"])


def _run_build(tmp_path, df):
    """Write df as Parquet, run matrix.run(config), return tmp_path (artifacts dir)."""
    df.to_parquet(tmp_path / "train.parquet", index=False)
    config = _make_config(tmp_path)
    run(config)
    return tmp_path


class TestMatrixShape:
    def test_shape_matches_unique_users_and_movies(self, tmp_path):
        df = _make_train_df([(1, 100, 1), (1, 101, -1), (1, 102, 2), (2, 100, 1), (2, 101, -2), (3, 102, 1)])
        arts = _run_build(tmp_path, df)
        R = load_npz(arts / "R_train.npz")
        assert R.shape == (3, 3)

    def test_single_user_single_movie(self, tmp_path):
        df = _make_train_df([(1, 100, 2)])
        arts = _run_build(tmp_path, df)
        R = load_npz(arts / "R_train.npz")
        assert R.shape == (1, 1)

    def test_nonzero_count_matches_interactions(self, tmp_path):
        df = _make_train_df([(1, 100, 1), (1, 101, -1), (2, 100, 2), (2, 102, -2)])
        arts = _run_build(tmp_path, df)
        R = load_npz(arts / "R_train.npz")
        assert R.nnz == 4


class TestMatrixValues:
    def test_preference_values_stored_correctly(self, tmp_path):
        df = _make_train_df([(1, 100, 2), (1, 101, -1), (2, 100, -2)])
        arts = _run_build(tmp_path, df)
        R = load_npz(arts / "R_train.npz")

        with open(arts / "mappings.json") as f:
            mappings = json.load(f)

        u1_idx = mappings["user_id_to_index"]["1"]
        u2_idx = mappings["user_id_to_index"]["2"]
        m100_idx = mappings["movie_id_to_index"]["100"]
        m101_idx = mappings["movie_id_to_index"]["101"]

        dense = R.toarray()
        assert dense[u1_idx, m100_idx] == pytest.approx(2.0)
        assert dense[u1_idx, m101_idx] == pytest.approx(-1.0)
        assert dense[u2_idx, m100_idx] == pytest.approx(-2.0)

    def test_matrix_dtype_is_float32(self, tmp_path):
        df = _make_train_df([(1, 100, 1)])
        arts = _run_build(tmp_path, df)
        R = load_npz(arts / "R_train.npz")
        assert R.dtype == np.float32


class TestMappings:
    def test_mappings_are_bijective(self, tmp_path):
        df = _make_train_df([(1, 100, 1), (2, 101, -1), (3, 102, 2)])
        arts = _run_build(tmp_path, df)

        with open(arts / "mappings.json") as f:
            mappings = json.load(f)

        mid_to_idx = mappings["movie_id_to_index"]
        idx_to_mid = mappings["index_to_movie_id"]
        for mid, idx in mid_to_idx.items():
            assert idx_to_mid[str(idx)] == int(mid)

    def test_user_mappings_cover_all_users(self, tmp_path):
        df = _make_train_df([(10, 100, 1), (20, 100, -1), (30, 101, 2)])
        arts = _run_build(tmp_path, df)

        with open(arts / "mappings.json") as f:
            mappings = json.load(f)

        assert {int(k) for k in mappings["user_id_to_index"]} == {10, 20, 30}

    def test_movie_mappings_cover_all_movies(self, tmp_path):
        df = _make_train_df([(1, 200, 1), (1, 300, -1), (2, 400, 2)])
        arts = _run_build(tmp_path, df)

        with open(arts / "mappings.json") as f:
            mappings = json.load(f)

        assert {int(k) for k in mappings["movie_id_to_index"]} == {200, 300, 400}

    def test_mappings_keys_are_native_int(self, tmp_path):
        df = _make_train_df([(1, 100, 1)])
        arts = _run_build(tmp_path, df)

        with open(arts / "mappings.json") as f:
            mappings = json.load(f)

        for section in mappings.values():
            for k, v in section.items():
                assert isinstance(k, str)
                assert isinstance(v, int)


class TestArtifactPersistence:
    def test_matrix_file_created(self, tmp_path):
        df = _make_train_df([(1, 100, 1)])
        arts = _run_build(tmp_path, df)
        assert (arts / "R_train.npz").exists()

    def test_mappings_file_created(self, tmp_path):
        df = _make_train_df([(1, 100, 1)])
        arts = _run_build(tmp_path, df)
        assert (arts / "mappings.json").exists()

    def test_mappings_has_all_three_sections(self, tmp_path):
        df = _make_train_df([(1, 100, 1)])
        arts = _run_build(tmp_path, df)

        with open(arts / "mappings.json") as f:
            mappings = json.load(f)

        assert "user_id_to_index" in mappings
        assert "movie_id_to_index" in mappings
        assert "index_to_movie_id" in mappings
