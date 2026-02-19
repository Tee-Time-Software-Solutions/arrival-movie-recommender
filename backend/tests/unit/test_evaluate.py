import numpy as np
import pytest

from movie_recommender.services.recommender.learning.evaluate import dcg_at_k


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
