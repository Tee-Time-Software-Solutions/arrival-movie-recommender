import pandas as pd
import pytest

from movie_recommender.services.recommender.pipeline.models.base.steps.preprocess_ratings import run
from movie_recommender.services.recommender.utils.schema import Config, DataConfig


def _make_config(tmp_path):
    return Config(data_dirs=DataConfig(
        source_dir=tmp_path,
        processed_dir=tmp_path,
        splits_dir=tmp_path,
        model_assets_dir=tmp_path,
    ))


class TestPreprocessRatingsOrchestration:
    def _run(self, tmp_path, csv_content):
        raw_path = tmp_path / "ratings.csv"
        raw_path.write_text(csv_content)
        config = _make_config(tmp_path)
        run(config)
        return tmp_path / "ratings_clean.parquet"

    def test_output_created(self, tmp_path):
        csv = "userId,movieId,rating,timestamp\n1,100,5.0,1000\n"
        out = self._run(tmp_path, csv)
        assert out.exists()

    def test_columns_correct(self, tmp_path):
        csv = "userId,movieId,rating,timestamp\n1,100,5.0,1000\n"
        out = self._run(tmp_path, csv)
        df = pd.read_parquet(out)
        assert list(df.columns) == ["user_id", "movie_id", "preference", "timestamp"]

    def test_preferences_mapped(self, tmp_path):
        csv = "userId,movieId,rating,timestamp\n1,100,5.0,1000\n2,200,1.0,2000\n"
        out = self._run(tmp_path, csv)
        df = pd.read_parquet(out)
        prefs = dict(zip(df["user_id"], df["preference"]))
        assert prefs[1] == 2   # rating 5.0 → preference +2
        assert prefs[2] == -2  # rating 1.0 → preference -2
