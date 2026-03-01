import numpy as np
import pytest

from movie_recommender.services.recommender.serving.online_updater import (
    update_user_vector,
)


def _vec(*vals):
    return np.array(vals, dtype=np.float32)


class TestSkipPreference:
    def test_skip_returns_copy_not_reference(self):
        user = _vec(1.0, 2.0, 3.0)
        movie = _vec(0.5, 0.5, 0.5)
        result = update_user_vector(user, movie, preference=0)
        assert result is not user
        np.testing.assert_array_equal(result, user)


class TestPositivePreference:
    def test_positive_moves_toward_movie(self):
        user = _vec(1.0, 0.0, 0.0)
        movie = _vec(0.0, 1.0, 0.0)
        result = update_user_vector(user, movie, preference=1, eta=0.1, norm_cap=100.0)
        assert result[1] > user[1]

    def test_negative_moves_away_from_movie(self):
        user = _vec(1.0, 1.0, 0.0)
        movie = _vec(0.0, 1.0, 0.0)
        result = update_user_vector(user, movie, preference=-1, eta=0.1, norm_cap=100.0)
        assert result[1] < user[1]


class TestPreferenceMagnitude:
    def test_larger_preference_gives_larger_step(self):
        user = _vec(1.0, 0.0, 0.0)
        movie = _vec(0.0, 1.0, 0.0)
        r1 = update_user_vector(user, movie, preference=1, eta=0.1, norm_cap=100.0)
        r2 = update_user_vector(user, movie, preference=2, eta=0.1, norm_cap=100.0)
        assert r2[1] > r1[1]


class TestEtaScaling:
    def test_eta_scales_step_size(self):
        user = _vec(1.0, 0.0, 0.0)
        movie = _vec(0.0, 1.0, 0.0)
        small = update_user_vector(user, movie, preference=1, eta=0.01, norm_cap=100.0)
        large = update_user_vector(user, movie, preference=1, eta=0.1, norm_cap=100.0)
        assert large[1] > small[1]


class TestNormCapping:
    def test_norm_capped_when_exceeds(self):
        user = _vec(9.0, 0.0, 0.0)
        movie = _vec(100.0, 0.0, 0.0)
        result = update_user_vector(user, movie, preference=1, eta=1.0, norm_cap=10.0)
        assert np.linalg.norm(result) == pytest.approx(10.0, abs=1e-5)

    def test_norm_capping_preserves_direction(self):
        user = _vec(5.0, 5.0, 0.0)
        movie = _vec(100.0, 100.0, 0.0)
        result = update_user_vector(user, movie, preference=1, eta=1.0, norm_cap=10.0)
        direction = result / np.linalg.norm(result)
        expected_dir = _vec(1.0, 1.0, 0.0)
        expected_dir = expected_dir / np.linalg.norm(expected_dir)
        np.testing.assert_allclose(direction, expected_dir, atol=1e-5)

    def test_large_preference_capped(self):
        user = _vec(1.0, 0.0, 0.0)
        movie = _vec(1.0, 0.0, 0.0)
        result = update_user_vector(user, movie, preference=1000, eta=1.0, norm_cap=10.0)
        assert np.linalg.norm(result) == pytest.approx(10.0, abs=1e-5)


class TestEdgeCases:
    def test_zero_movie_vector_returns_unchanged(self):
        user = _vec(1.0, 2.0, 3.0)
        movie = _vec(0.0, 0.0, 0.0)
        result = update_user_vector(user, movie, preference=1)
        np.testing.assert_array_almost_equal(result, user)

    def test_output_dtype_is_float32(self):
        user = np.array([1.0, 2.0], dtype=np.float64)
        movie = np.array([0.5, 0.5], dtype=np.float64)
        result = update_user_vector(user, movie, preference=1)
        assert result.dtype == np.float32
