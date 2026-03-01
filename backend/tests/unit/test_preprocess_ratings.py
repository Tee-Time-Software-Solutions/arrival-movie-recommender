import pandas as pd
import pytest
from unittest.mock import patch

from movie_recommender.services.recommender.data_processing.preprocessing.preprocess_ratings import (
    map_rating_to_bucket,
    bucket_to_preference,
    preprocess_ratings,
)

_MODULE = "movie_recommender.services.recommender.data_processing.preprocessing.preprocess_ratings"


class TestMapRatingToBucket:
    def test_below_2_5_is_bucket_1(self):
        assert map_rating_to_bucket(0.5) == 1
        assert map_rating_to_bucket(2.0) == 1
        assert map_rating_to_bucket(2.4) == 1

    def test_at_2_5_is_bucket_2(self):
        assert map_rating_to_bucket(2.5) == 2

    def test_below_3_5_is_bucket_2(self):
        assert map_rating_to_bucket(3.0) == 2
        assert map_rating_to_bucket(3.4) == 2

    def test_at_3_5_is_bucket_3(self):
        assert map_rating_to_bucket(3.5) == 3

    def test_below_4_5_is_bucket_3(self):
        assert map_rating_to_bucket(4.0) == 3
        assert map_rating_to_bucket(4.4) == 3

    def test_at_4_5_is_bucket_4(self):
        assert map_rating_to_bucket(4.5) == 4

    def test_max_rating_is_bucket_4(self):
        assert map_rating_to_bucket(5.0) == 4

    def test_extremes(self):
        assert map_rating_to_bucket(0.5) == 1
        assert map_rating_to_bucket(5.0) == 4


class TestBucketToPreference:
    def test_bucket_1(self):
        assert bucket_to_preference(1) == -2

    def test_bucket_2(self):
        assert bucket_to_preference(2) == -1

    def test_bucket_3(self):
        assert bucket_to_preference(3) == 1

    def test_bucket_4(self):
        assert bucket_to_preference(4) == 2

    def test_invalid_bucket_raises(self):
        with pytest.raises(KeyError):
            bucket_to_preference(5)


class TestPreprocessRatingsOrchestration:
    """Integration-style tests for the preprocess_ratings() orchestration wrapper."""

    def _run(self, tmp_path, csv_content):
        raw_path = tmp_path / "ratings.csv"
        raw_path.write_text(csv_content)
        out_path = tmp_path / "interactions_clean.parquet"

        with patch(f"{_MODULE}.RAW_PATH", raw_path), \
             patch(f"{_MODULE}.PROCESSED_PATH", out_path):
            preprocess_ratings()
        return out_path

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
        assert prefs[1] == 2    # rating 5.0 → bucket 4 → preference +2
        assert prefs[2] == -2   # rating 1.0 → bucket 1 → preference -2
