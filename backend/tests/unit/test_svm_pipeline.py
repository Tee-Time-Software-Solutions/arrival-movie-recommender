import json

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import load_npz

from movie_recommender.services.recommender.pipeline.offline.models.svm import (
    main as svm_main,
)
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


def _make_config_with_ratio(tmp_path, ratio: float) -> Config:
    return Config(
        data_dirs=DataConfig(
            source_dir=tmp_path / "source",
            processed_dir=tmp_path,
            splits_dir=tmp_path,
            model_assets_dir=tmp_path,
        ),
        models=ModelsConfig(
            svm=SVMConfig(
                c=1.0,
                max_iter=300,
                negative_sampling_ratio=ratio,
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
        metadata_indices = []
        for per_movie in mappings["movie_metadata_feature_indices"].values():
            metadata_indices.extend(per_movie)
        assert metadata_indices
        assert max(metadata_indices) < (
            len(mappings["genre_to_index"]) + len(mappings["year_bucket_to_index"])
        )

    def test_low_negative_ratio_still_creates_negative_labels(self, tmp_path):
        _write_synthetic_inputs(tmp_path)
        config = _make_config_with_ratio(tmp_path, ratio=0.2)

        svm_data.run(config)

        labels = np.load(tmp_path / "svm_train_labels.npy")
        assert set(np.unique(labels).tolist()) == {0, 1}


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


class TestSvmMainFallback:
    def test_uses_existing_artifacts_when_source_csvs_missing(
        self, tmp_path, monkeypatch
    ):
        config = _make_config_with_ratio(tmp_path, ratio=1.0)
        config.data_dirs.source_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            {
                "movie_id": [10],
                "title": ["A"],
                "release_year": [2000],
                "genres": ["Drama"],
            }
        ).to_parquet(tmp_path / "movies_filtered.parquet", index=False)
        pd.DataFrame(
            {"user_id": [1], "movie_id": [10], "preference": [1], "timestamp": [1]}
        ).to_parquet(tmp_path / "train.parquet", index=False)
        pd.DataFrame(
            {"user_id": [1], "movie_id": [10], "preference": [1], "timestamp": [2]}
        ).to_parquet(tmp_path / "val.parquet", index=False)

        calls = {"preprocess_movies": 0, "data": 0, "trainer": 0, "evaluator": 0}

        monkeypatch.setattr(svm_main, "load_config", lambda: config)
        monkeypatch.setattr(
            svm_main.preprocess_movies,
            "run",
            lambda _config: calls.__setitem__(
                "preprocess_movies", calls["preprocess_movies"] + 1
            ),
        )
        monkeypatch.setattr(
            svm_main.svm_data,
            "run",
            lambda _config: calls.__setitem__("data", calls["data"] + 1),
        )
        monkeypatch.setattr(
            svm_main.trainer,
            "run",
            lambda _config: calls.__setitem__("trainer", calls["trainer"] + 1),
        )
        monkeypatch.setattr(
            svm_main.evaluator,
            "run",
            lambda _config: calls.__setitem__("evaluator", calls["evaluator"] + 1),
        )

        svm_main.SVMPipeline().run_pipeline()

        assert calls["preprocess_movies"] == 0
        assert calls["data"] == 1
        assert calls["trainer"] == 1
        assert calls["evaluator"] == 1

    def test_raises_actionable_error_when_no_source_or_base_artifacts(
        self, tmp_path, monkeypatch
    ):
        config = _make_config_with_ratio(tmp_path, ratio=1.0)
        config.data_dirs.source_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(svm_main, "load_config", lambda: config)

        with pytest.raises(
            FileNotFoundError, match="MovieLens source files were not found"
        ):
            svm_main.SVMPipeline().run_pipeline()
