import json

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

from movie_recommender.services.recommender.pipeline.offline.models.item_cf.steps.train_item_cf import (
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
        models=ModelsConfig(
            item_cf=ItemCFConfig(
                similarity="cosine",
                top_k_neighbors=1,
                min_similarity=0.1,
                use_positive_only=True,
                normalize_scores=True,
            )
        ),
    )


def test_item_cf_trainer_writes_similarity_and_model_info(tmp_path):
    matrix = csr_matrix(
        np.array(
            [
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    save_npz(tmp_path / "item_cf_train_matrix.npz", matrix)

    run(_make_config(tmp_path))

    similarity = load_npz(tmp_path / "item_cf_similarity.npz").tocsr()
    assert similarity.shape == (3, 3)
    assert np.allclose(similarity.diagonal(), 0.0)
    assert max(similarity.getrow(row_idx).nnz for row_idx in range(3)) <= 1

    with open(tmp_path / "item_cf_model_info.json") as file_obj:
        info = json.load(file_obj)

    assert info["model"] == "item_cf"
    assert info["similarity"] == "cosine"
    assert info["top_k_neighbors"] == 1
