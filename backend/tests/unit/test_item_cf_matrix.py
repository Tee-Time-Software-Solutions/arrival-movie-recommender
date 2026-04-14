import json

import pandas as pd
from scipy.sparse import load_npz

from movie_recommender.services.recommender.pipeline.offline.models.item_cf.steps.matrix import (
    run,
)
from movie_recommender.services.recommender.utils.schema import Config, DataConfig


def _make_config(tmp_path):
    return Config(
        data_dirs=DataConfig(
            source_dir=tmp_path,
            processed_dir=tmp_path,
            splits_dir=tmp_path,
            model_assets_dir=tmp_path,
        )
    )


def test_item_cf_matrix_builds_sparse_matrix_and_mappings(tmp_path):
    train_df = pd.DataFrame(
        [
            (1, 100, 2.0),
            (1, 101, -1.0),
            (2, 100, 1.0),
            (3, 102, 2.0),
        ],
        columns=["user_id", "movie_id", "preference"],
    )
    train_df.to_parquet(tmp_path / "train.parquet", index=False)

    run(_make_config(tmp_path))

    matrix = load_npz(tmp_path / "item_cf_train_matrix.npz")
    assert matrix.shape == (3, 3)
    assert matrix.nnz == 4

    with open(tmp_path / "item_cf_mappings.json") as file_obj:
        mappings = json.load(file_obj)

    assert set(mappings) == {
        "user_id_to_index",
        "movie_id_to_index",
        "index_to_user_id",
        "index_to_movie_id",
    }
    assert set(map(int, mappings["movie_id_to_index"].keys())) == {100, 101, 102}
