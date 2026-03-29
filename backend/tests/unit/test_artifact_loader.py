import pytest
from unittest.mock import patch
from pathlib import Path

from movie_recommender.services.recommender.serving.artifact_loader import (
    _ensure_artifact_paths_exist,
)


class TestEnsureArtifactPathsExist:
    def test_all_present_no_error(self, tmp_path):
        paths = [tmp_path / f"file_{i}.npy" for i in range(4)]
        for p in paths:
            p.touch()

        with patch(
            "movie_recommender.services.recommender.serving.artifact_loader.MOVIE_EMBEDDINGS_PATH",
            paths[0],
        ), patch(
            "movie_recommender.services.recommender.serving.artifact_loader.USER_EMBEDDINGS_PATH",
            paths[1],
        ), patch(
            "movie_recommender.services.recommender.serving.artifact_loader.MAPPINGS_PATH",
            paths[2],
        ), patch(
            "movie_recommender.services.recommender.serving.artifact_loader.MOVIES_FILTERED_PATH",
            paths[3],
        ):
            _ensure_artifact_paths_exist()  # should not raise

    def test_missing_file_raises(self, tmp_path):
        existing = tmp_path / "exists.npy"
        existing.touch()
        missing = tmp_path / "missing.npy"

        with patch(
            "movie_recommender.services.recommender.serving.artifact_loader.MOVIE_EMBEDDINGS_PATH",
            existing,
        ), patch(
            "movie_recommender.services.recommender.serving.artifact_loader.USER_EMBEDDINGS_PATH",
            missing,
        ), patch(
            "movie_recommender.services.recommender.serving.artifact_loader.MAPPINGS_PATH",
            existing,
        ), patch(
            "movie_recommender.services.recommender.serving.artifact_loader.MOVIES_FILTERED_PATH",
            existing,
        ):
            with pytest.raises(FileNotFoundError, match="missing.npy"):
                _ensure_artifact_paths_exist()

    def test_multiple_missing_lists_all(self, tmp_path):
        existing = tmp_path / "exists.npy"
        existing.touch()
        missing1 = tmp_path / "missing1.npy"
        missing2 = tmp_path / "missing2.npy"

        with patch(
            "movie_recommender.services.recommender.serving.artifact_loader.MOVIE_EMBEDDINGS_PATH",
            missing1,
        ), patch(
            "movie_recommender.services.recommender.serving.artifact_loader.USER_EMBEDDINGS_PATH",
            missing2,
        ), patch(
            "movie_recommender.services.recommender.serving.artifact_loader.MAPPINGS_PATH",
            existing,
        ), patch(
            "movie_recommender.services.recommender.serving.artifact_loader.MOVIES_FILTERED_PATH",
            existing,
        ):
            with pytest.raises(FileNotFoundError) as exc_info:
                _ensure_artifact_paths_exist()
            assert "missing1.npy" in str(exc_info.value)
            assert "missing2.npy" in str(exc_info.value)

    def test_partial_missing_lists_only_missing(self, tmp_path):
        paths = [tmp_path / f"file_{i}.npy" for i in range(4)]
        paths[0].touch()
        paths[1].touch()
        paths[2].touch()
        # paths[3] is missing

        with patch(
            "movie_recommender.services.recommender.serving.artifact_loader.MOVIE_EMBEDDINGS_PATH",
            paths[0],
        ), patch(
            "movie_recommender.services.recommender.serving.artifact_loader.USER_EMBEDDINGS_PATH",
            paths[1],
        ), patch(
            "movie_recommender.services.recommender.serving.artifact_loader.MAPPINGS_PATH",
            paths[2],
        ), patch(
            "movie_recommender.services.recommender.serving.artifact_loader.MOVIES_FILTERED_PATH",
            paths[3],
        ):
            with pytest.raises(FileNotFoundError) as exc_info:
                _ensure_artifact_paths_exist()
            msg = str(exc_info.value)
            assert "file_3.npy" in msg
            assert "file_0.npy" not in msg
            assert "file_1.npy" not in msg
            assert "file_2.npy" not in msg
