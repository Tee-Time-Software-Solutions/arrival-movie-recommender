import pytest
from pydantic import ValidationError

from movie_recommender.schemas.requests.interactions import (
    RegisteredFeedback,
    SwipeAction,
    SwipeRequest,
)


class TestSwipeRequest:
    def test_defaults(self):
        req = SwipeRequest(action_type=SwipeAction.LIKE)
        assert req.is_supercharged is False

    def test_supercharged_like(self):
        req = SwipeRequest(action_type="like", is_supercharged=True)
        assert req.action_type == SwipeAction.LIKE
        assert req.is_supercharged is True

    def test_invalid_action_raises(self):
        with pytest.raises(ValidationError):
            SwipeRequest(action_type="love")


class TestRegisteredFeedback:
    def test_feedback_roundtrip(self):
        fb = RegisteredFeedback(
            interaction_id=42,
            movie_id=1,
            action_type=SwipeAction.DISLIKE,
            is_supercharged=False,
            registered=True,
        )
        assert fb.interaction_id == 42
        assert fb.registered is True
