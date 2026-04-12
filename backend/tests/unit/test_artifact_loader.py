import pytest
from unittest.mock import patch, MagicMock

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
