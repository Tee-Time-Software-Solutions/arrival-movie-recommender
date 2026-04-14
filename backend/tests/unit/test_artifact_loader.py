import pytest
from unittest.mock import patch, MagicMock
import json

import numpy as np
import pandas as pd

from movie_recommender.services.recommender.pipeline.online.artifacts import (
    load_model_artifacts,
)


def _make_config(tmp_path):
    mock_config = MagicMock()
    mock_config.data_dirs.model_assets_dir = tmp_path / "assets"
    mock_config.data_dirs.processed_dir = tmp_path / "processed"
    return mock_config


class TestLoadModelArtifacts:
    def test_raises_when_all_files_missing(self, tmp_path):
        config = _make_config(tmp_path)
        with patch(
            "movie_recommender.services.recommender.pipeline.online.artifacts.load_config",
            return_value=config,
        ):
            with pytest.raises(FileNotFoundError):
                load_model_artifacts()

    def test_error_message_lists_missing_files(self, tmp_path):
        config = _make_config(tmp_path)
        with patch(
            "movie_recommender.services.recommender.pipeline.online.artifacts.load_config",
            return_value=config,
        ):
            with pytest.raises(FileNotFoundError) as exc_info:
                load_model_artifacts()
        msg = str(exc_info.value)
        assert "movie_embeddings.npy" in msg
        assert "mappings.json" in msg

    def test_partial_missing_lists_only_missing_files(self, tmp_path):
        config = _make_config(tmp_path)
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        (assets_dir / "movie_embeddings.npy").touch()
        (assets_dir / "user_embeddings.npy").touch()
        # mappings.json and movies_filtered.parquet intentionally absent

        with patch(
            "movie_recommender.services.recommender.pipeline.online.artifacts.load_config",
            return_value=config,
        ):
            with pytest.raises(FileNotFoundError) as exc_info:
                load_model_artifacts()
        msg = str(exc_info.value)
        assert "mappings.json" in msg
        assert "movies_filtered.parquet" in msg
        assert "movie_embeddings.npy" not in msg
        assert "user_embeddings.npy" not in msg

    def test_error_includes_run_pipeline_hint(self, tmp_path):
        config = _make_config(tmp_path)
        with patch(
            "movie_recommender.services.recommender.pipeline.online.artifacts.load_config",
            return_value=config,
        ):
            with pytest.raises(FileNotFoundError) as exc_info:
                load_model_artifacts()
        assert "offline pipeline" in str(exc_info.value)

    def test_loads_movie_genres_from_processed_movies(self, tmp_path):
        config = _make_config(tmp_path)
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        np.save(assets_dir / "movie_embeddings.npy", np.array([[1.0]], dtype=np.float32))
        np.save(assets_dir / "user_embeddings.npy", np.array([[1.0]], dtype=np.float32))
        (assets_dir / "mappings.json").write_text(
            json.dumps(
                {
                    "user_id_to_index": {"1": 0},
                    "movie_id_to_index": {"100": 0},
                    "index_to_movie_id": {"0": 100},
                }
            )
        )
        pd.DataFrame(
            {
                "movie_id": [100, 101, 102],
                "title": ["Action Movie", "No Genre Movie", "Sci-Fi Comedy"],
                "genres": ["Action|Thriller", "", None],
            }
        ).to_parquet(processed_dir / "movies_filtered.parquet", index=False)

        with patch(
            "movie_recommender.services.recommender.pipeline.online.artifacts.load_config",
            return_value=config,
        ):
            artifacts = load_model_artifacts()

        assert artifacts.movie_id_to_genres[100] == ["Action", "Thriller"]
        assert artifacts.movie_id_to_genres[101] == []
        assert artifacts.movie_id_to_genres[102] == []
