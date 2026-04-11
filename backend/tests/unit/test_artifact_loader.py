import pytest
from pathlib import Path
from unittest.mock import patch

from movie_recommender.services.recommender.serving import artifact_loader as al
from movie_recommender.services.recommender.serving.artifact_loader import (
    _ensure_artifact_paths_exist,
)


class TestEnsureArtifactPathsExist:
    def _touch_embeddings_and_mappings(self, emb_root: Path):
        (emb_root / "movie_embeddings.npy").touch()
        (emb_root / "user_embeddings.npy").touch()
        (emb_root / "mappings.json").touch()

    def test_all_present_no_error(self, tmp_path):
        emb_root = tmp_path / "emb"
        emb_root.mkdir()
        self._touch_embeddings_and_mappings(emb_root)
        (tmp_path / "movies_filtered.parquet").touch()

        with (
            patch.object(al, "artifacts_dir", lambda: emb_root),
            patch.object(al, "DATA_PROCESSED", tmp_path),
        ):
            _ensure_artifact_paths_exist()

    def test_missing_file_raises(self, tmp_path):
        emb_root = tmp_path / "emb"
        emb_root.mkdir()
        (emb_root / "movie_embeddings.npy").touch()
        (emb_root / "mappings.json").touch()
        (tmp_path / "movies_filtered.parquet").touch()

        with (
            patch.object(al, "artifacts_dir", lambda: emb_root),
            patch.object(al, "DATA_PROCESSED", tmp_path),
        ):
            with pytest.raises(FileNotFoundError, match="user_embeddings"):
                _ensure_artifact_paths_exist()

    def test_multiple_missing_lists_all(self, tmp_path):
        emb_root = tmp_path / "emb"
        emb_root.mkdir()
        (tmp_path / "movies_filtered.parquet").touch()

        with (
            patch.object(al, "artifacts_dir", lambda: emb_root),
            patch.object(al, "DATA_PROCESSED", tmp_path),
        ):
            with pytest.raises(FileNotFoundError) as exc_info:
                _ensure_artifact_paths_exist()
            msg = str(exc_info.value)
            assert "movie_embeddings" in msg or "user_embeddings" in msg

    def test_partial_missing_lists_only_missing(self, tmp_path):
        emb_root = tmp_path / "emb"
        emb_root.mkdir()
        self._touch_embeddings_and_mappings(emb_root)

        with (
            patch.object(al, "artifacts_dir", lambda: emb_root),
            patch.object(al, "DATA_PROCESSED", tmp_path),
        ):
            with pytest.raises(FileNotFoundError) as exc_info:
                _ensure_artifact_paths_exist()
            assert "movies_filtered" in str(exc_info.value)
