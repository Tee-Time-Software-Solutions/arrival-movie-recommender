import numpy as np
import pytest

from movie_recommender.services.recommender.main import _to_int_user_id, _top_n_indices


class TestToIntUserId:
    def test_valid_int_string(self):
        assert _to_int_user_id("42") == 42

    def test_non_numeric_string(self):
        assert _to_int_user_id("abc") is None

    def test_none_input(self):
        assert _to_int_user_id(None) is None

    def test_float_string(self):
        assert _to_int_user_id("3.14") is None


class TestTopNIndices:
    def test_normal_case_returns_sorted_indices(self):
        scores = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        result = _top_n_indices(scores, 3)
        assert result == [3, 4, 1]

    def test_n_zero_returns_empty(self):
        scores = np.array([1.0, 2.0, 3.0])
        assert _top_n_indices(scores, 0) == []

    def test_empty_scores_returns_empty(self):
        scores = np.array([])
        assert _top_n_indices(scores, 5) == []

    def test_n_greater_than_length_returns_all_sorted(self):
        scores = np.array([3.0, 1.0, 2.0])
        result = _top_n_indices(scores, 10)
        assert result == [0, 2, 1]
