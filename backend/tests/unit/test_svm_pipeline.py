import json

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from movie_recommender.services.recommender.pipeline.offline.models.svm.steps import (
    data as svm_data,
)
from movie_recommender.services.recommender.pipeline.offline.models.svm.steps import (
    evaluator as svm_evaluator,
)
from movie_recommender.services.recommender.pipeline.offline.models.svm.steps import (
    trainer as svm_trainer,
)
from movie_recommender.services.recommender.utils.schema import (
    Config,
    DataConfig,
    ModelsConfig,
    SVMConfig,
)


def _make_config(tmp_path) -> Config:
    return Config(
        data_dirs=DataConfig(
            source_dir=tmp_path,
            processed_dir=tmp_path,
            splits_dir=tmp_path,
            model_assets_dir=tmp_path,
        ),
        models=ModelsConfig(
            svm=SVMConfig(
                c=1.0,
                max_iter=300,
                negative_sampling_ratio=2.0,
                random_state=7,
                use_metadata_features=True,
                release_year_bucket_size=10,
            )
        ),
    )


def _write_synthetic_inputs(tmp_path) -> None:
    train_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "movie_id": [10, 20, 20, 30, 30, 40],
            "preference": [1, 2, 1, 1, 1, 1],
            "timestamp": [1, 2, 1, 2, 1, 2],
        }
    )
    val_df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 1],
            "movie_id": [30, 40, 50, 60],
            "preference": [1, 1, 1, -1],
            "timestamp": [3, 3, 3, 4],
        }
    )
    movies_df = pd.DataFrame(
        {
            "movie_id": [10, 20, 30, 40, 50, 60],
            "title": ["A", "B", "C", "D", "E", "F"],
            "release_year": [1999, 2001, 2005, 2011, 2018, 2022],
            "genres": [
                "Action|Sci-Fi",
                "Drama",
                "Action|Drama",
                "Comedy",
                "Sci-Fi",
                "Drama|Thriller",
            ],
        }
    )
    train_df.to_parquet(tmp_path / "train.parquet", index=False)
    val_df.to_parquet(tmp_path / "val.parquet", index=False)
    movies_df.to_parquet(tmp_path / "movies_filtered.parquet", index=False)


class TestSvmDataStep:
    def test_creates_artifacts_and_valid_shapes(self, tmp_path):
        _write_synthetic_inputs(tmp_path)
        config = _make_config(tmp_path)

        svm_data.run(config)

        assert (tmp_path / "svm_train_features.npz").exists()
        assert (tmp_path / "svm_train_labels.npy").exists()
        assert (tmp_path / "svm_feature_mappings.json").exists()

        features = load_npz(tmp_path / "svm_train_features.npz")
        labels = np.load(tmp_path / "svm_train_labels.npy")
        with open(tmp_path / "svm_feature_mappings.json") as handle:
            mappings = json.load(handle)

        assert features.shape[0] == len(labels)
        assert features.shape[0] > 0
        assert set(np.unique(labels).tolist()) == {0, 1}
        assert mappings["feature_layout"]["num_features"] == features.shape[1]
        assert len(mappings["user_id_to_index"]) == 3
        assert len(mappings["movie_id_to_index"]) == 6


class TestSvmTrainerStep:
    def test_trainer_writes_model_artifacts(self, tmp_path):
        _write_synthetic_inputs(tmp_path)
        config = _make_config(tmp_path)
        svm_data.run(config)

        svm_trainer.run(config)

        assert (tmp_path / "svm_linear_model.joblib").exists()
        assert (tmp_path / "svm_model_info.json").exists()

        with open(tmp_path / "svm_model_info.json") as handle:
            info = json.load(handle)
        assert info["model"] == "linear_svm"
        assert info["num_samples"] > 0
        assert info["num_features"] > 0


class TestSvmEvaluatorStep:
    def test_evaluator_writes_metrics_report(self, tmp_path):
        _write_synthetic_inputs(tmp_path)
        config = _make_config(tmp_path)
        svm_data.run(config)
        svm_trainer.run(config)

        svm_evaluator.run(config)

        metrics_path = tmp_path / "svm_metrics.json"
        assert metrics_path.exists()

        with open(metrics_path) as handle:
            report = json.load(handle)
        assert report["model"] == "svm"
        assert report["k"] == 10
        assert "recall@10" in report["metrics"]
        assert "precision@10" in report["metrics"]
        assert "ndcg@10" in report["metrics"]
        assert report["num_users_evaluated"] >= 1
