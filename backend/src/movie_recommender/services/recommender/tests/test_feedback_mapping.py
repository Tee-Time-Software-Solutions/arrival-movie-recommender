from movie_recommender.schemas.interactions import SwipeAction
from movie_recommender.services.recommender.serving.feedback_mapping import (
    swipe_to_preference,
)


def test_dislike_maps_to_negative_one():
    assert swipe_to_preference(SwipeAction.DISLIKE, False) == -1


def test_supercharged_dislike_maps_to_negative_two():
    assert swipe_to_preference(SwipeAction.DISLIKE, True) == -2


def test_like_maps_to_positive_one():
    assert swipe_to_preference(SwipeAction.LIKE, False) == 1


def test_supercharged_like_maps_to_positive_two():
    assert swipe_to_preference(SwipeAction.LIKE, True) == 2


def test_skip_maps_to_zero():
    assert swipe_to_preference(SwipeAction.SKIP, False) == 0
