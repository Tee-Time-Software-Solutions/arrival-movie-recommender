import pytest

from movie_recommender.services.recommender.data_processing.preprocessing.preprocess_ratings import (
    map_rating_to_bucket,
    bucket_to_preference,
)


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
