import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from movie_recommender.services.recommender.learning.evaluate import dcg_at_k, evaluate

_MODULE = "movie_recommender.services.recommender.learning.evaluate"


class TestDcgAtK:
    def test_all_relevant(self):
        relevance = [1, 1, 1]
        result = dcg_at_k(relevance)
        expected = 1/np.log2(2) + 1/np.log2(3) + 1/np.log2(4)
        assert result == pytest.approx(expected)

    def test_no_relevant_items(self):
        assert dcg_at_k([0, 0, 0]) == 0.0

    def test_single_relevant_at_position_zero(self):
        result = dcg_at_k([1])
        expected = 1 / np.log2(2)
        assert result == pytest.approx(expected)

    def test_later_items_score_less(self):
        dcg_first = dcg_at_k([1, 0, 0])
        dcg_last = dcg_at_k([0, 0, 1])
        assert dcg_first > dcg_last

    def test_empty_relevance(self):
        assert dcg_at_k([]) == 0


class TestEvaluateOrchestration:
    """Integration-style tests for the evaluate() orchestration wrapper."""

    def _setup_artifacts(self, tmp_path, extra_val_users=None):
        n_users, n_movies, dim = 3, 5, 4
        rng = np.random.default_rng(42)

        user_emb = rng.standard_normal((n_users, dim)).astype(np.float32)
        movie_emb = rng.standard_normal((n_movies, dim)).astype(np.float32)

        np.save(tmp_path / "user_embeddings.npy", user_emb)
        np.save(tmp_path / "movie_embeddings.npy", movie_emb)

        mappings = {
            "user_id_to_index": {"1": 0, "2": 1, "3": 2},
            "movie_id_to_index": {"10": 0, "20": 1, "30": 2, "40": 3, "50": 4},
            "index_to_movie_id": {"0": 10, "1": 20, "2": 30, "3": 40, "4": 50},
        }
        with open(tmp_path / "mappings.json", "w") as f:
            json.dump(mappings, f)

        train_df = pd.DataFrame({"user_id": [1, 2, 3], "movie_id": [10, 20, 30]})
        train_df.to_parquet(tmp_path / "train.parquet", index=False)

        val_users = [1, 2, 3]
        val_movies = [40, 50, 10]
        if extra_val_users:
            for uid, mid in extra_val_users:
                val_users.append(uid)
                val_movies.append(mid)

        val_df = pd.DataFrame({"user_id": val_users, "movie_id": val_movies})
        val_df.to_parquet(tmp_path / "val.parquet", index=False)

    def _run(self, tmp_path):
        with patch(f"{_MODULE}.USER_EMB_PATH", tmp_path / "user_embeddings.npy"), \
             patch(f"{_MODULE}.MOVIE_EMB_PATH", tmp_path / "movie_embeddings.npy"), \
             patch(f"{_MODULE}.MAPPINGS_PATH", tmp_path / "mappings.json"), \
             patch(f"{_MODULE}.TRAIN_PATH", tmp_path / "train.parquet"), \
             patch(f"{_MODULE}.VAL_PATH", tmp_path / "val.parquet"), \
             patch(f"{_MODULE}.K", 3):
            evaluate()

    def test_runs_without_error(self, tmp_path):
        self._setup_artifacts(tmp_path)
        self._run(tmp_path)

    def test_skips_unknown_val_user(self, tmp_path):
        self._setup_artifacts(tmp_path, extra_val_users=[(999, 40)])
        self._run(tmp_path)  # user 999 not in mappings → silently skipped
