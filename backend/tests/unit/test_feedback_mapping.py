import pytest

from movie_recommender.schemas.interactions import SwipeAction
from movie_recommender.services.recommender.serving.feedback_mapping import (
    swipe_to_preference,
)


class TestLikePreference:
    def test_like_maps_to_positive_one(self):
        assert swipe_to_preference(SwipeAction.LIKE, False) == 1

    def test_supercharged_like_maps_to_positive_two(self):
        assert swipe_to_preference(SwipeAction.LIKE, True) == 2

    def test_like_return_type_is_int(self):
        result = swipe_to_preference(SwipeAction.LIKE, False)
        assert isinstance(result, int)


class TestDislikePreference:
    def test_dislike_maps_to_negative_one(self):
        assert swipe_to_preference(SwipeAction.DISLIKE, False) == -1

    def test_supercharged_dislike_maps_to_negative_two(self):
        assert swipe_to_preference(SwipeAction.DISLIKE, True) == -2

    def test_dislike_return_type_is_int(self):
        result = swipe_to_preference(SwipeAction.DISLIKE, False)
        assert isinstance(result, int)


class TestSkipPreference:
    def test_skip_maps_to_zero(self):
        assert swipe_to_preference(SwipeAction.SKIP, False) == 0

    def test_skip_supercharged_still_zero(self):
        """Supercharged flag is irrelevant for skip — should still return 0."""
        assert swipe_to_preference(SwipeAction.SKIP, True) == 0

    def test_skip_return_type_is_int(self):
        result = swipe_to_preference(SwipeAction.SKIP, False)
        assert isinstance(result, int)


class TestEdgeCases:
    def test_all_actions_return_int(self):
        for action in SwipeAction:
            for supercharged in [True, False]:
                result = swipe_to_preference(action, supercharged)
                assert isinstance(result, int), f"Failed for {action}, supercharged={supercharged}"

    def test_preference_symmetry(self):
        """Like and dislike should be symmetric in magnitude."""
        like = swipe_to_preference(SwipeAction.LIKE, False)
        dislike = swipe_to_preference(SwipeAction.DISLIKE, False)
        assert abs(like) == abs(dislike)

    def test_supercharged_symmetry(self):
        """Supercharged like and dislike should be symmetric in magnitude."""
        like = swipe_to_preference(SwipeAction.LIKE, True)
        dislike = swipe_to_preference(SwipeAction.DISLIKE, True)
        assert abs(like) == abs(dislike)

    def test_supercharged_magnitude_greater_than_regular(self):
        """Supercharged should always have greater magnitude than regular."""
        assert abs(swipe_to_preference(SwipeAction.LIKE, True)) > abs(swipe_to_preference(SwipeAction.LIKE, False))
        assert abs(swipe_to_preference(SwipeAction.DISLIKE, True)) > abs(swipe_to_preference(SwipeAction.DISLIKE, False))
