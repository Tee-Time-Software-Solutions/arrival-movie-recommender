import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

from movie_recommender.services.recommender.pipeline.offline.models.item_cf.steps.metrics import (
    run,
)
from movie_recommender.services.recommender.utils.schema import (
    Config,
    DataConfig,
    ItemCFConfig,
    ModelsConfig,
)


def _make_config(tmp_path):
    return Config(
        data_dirs=DataConfig(
            source_dir=tmp_path,
            processed_dir=tmp_path,
            splits_dir=tmp_path,
            model_assets_dir=tmp_path,
        ),
        models=ModelsConfig(item_cf=ItemCFConfig()),
    )


def test_item_cf_evaluator_writes_metrics_json(tmp_path):
    train_df = pd.DataFrame(
        [(1, 10, 1.0), (2, 30, 1.0)],
        columns=["user_id", "movie_id", "preference"],
    )
    val_df = pd.DataFrame(
        [(1, 30, 1.0), (2, 20, 1.0)],
        columns=["user_id", "movie_id", "preference"],
    )
    train_df.to_parquet(tmp_path / "train.parquet", index=False)
    val_df.to_parquet(tmp_path / "val.parquet", index=False)

    train_matrix = csr_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    similarity = csr_matrix(
        np.array(
            [
                [0.0, 0.1, 0.9],
                [0.1, 0.0, 0.9],
                [0.9, 0.9, 0.0],
            ],
            dtype=np.float32,
        )
    )
    save_npz(tmp_path / "item_cf_train_matrix.npz", train_matrix)
    save_npz(tmp_path / "item_cf_similarity.npz", similarity)

    with open(tmp_path / "item_cf_mappings.json", "w") as file_obj:
        json.dump(
            {
                "user_id_to_index": {"1": 0, "2": 1},
                "movie_id_to_index": {"10": 0, "20": 1, "30": 2},
                "index_to_user_id": {"0": 1, "1": 2},
                "index_to_movie_id": {"0": 10, "1": 20, "2": 30},
            },
            file_obj,
        )

    run(_make_config(tmp_path))

    report_path = tmp_path / "item_cf_metrics.json"
    assert report_path.exists()

    with open(report_path) as file_obj:
        report = json.load(file_obj)

    assert report["model"] == "item_cf"
    assert report["k"] == 10
    assert report["num_users_evaluated"] >= 1
    assert "precision@10" in report["metrics"]
    assert "recall@10" in report["metrics"]
    assert "ndcg@10" in report["metrics"]
