import pytest

from movie_recommender.schemas.requests.interactions import SwipeAction
from movie_recommender.services.recommender.pipeline.online.learning.adaptive import (
    FEEDBACK_COUNT_TTL_SECONDS,
    adaptive_learning_rate,
    counts_for_adaptation,
    feedback_count_key,
    get_feedback_count,
    increment_feedback_count,
)


class TestAdaptiveLearningRate:
    def test_strength_zero_keeps_base_learning_rate(self):
        assert adaptive_learning_rate(0.05, feedback_count=0, strength=0.0) == 0.05

    def test_new_user_gets_larger_learning_rate(self):
        fresh = adaptive_learning_rate(0.05, feedback_count=0, strength=0.2)
        mature = adaptive_learning_rate(0.05, feedback_count=100, strength=0.2)
        assert fresh > mature
        assert fresh > 0.05

    def test_learning_rate_decays_toward_base(self):
        rate = adaptive_learning_rate(0.05, feedback_count=10_000, strength=0.2)
        assert rate == pytest.approx(0.05, rel=0.03)


class TestFeedbackCounting:
    def test_skip_does_not_count_for_adaptation(self):
        assert counts_for_adaptation(SwipeAction.SKIP) is False

    def test_like_and_dislike_count_for_adaptation(self):
        assert counts_for_adaptation(SwipeAction.LIKE) is True
        assert counts_for_adaptation(SwipeAction.DISLIKE) is True

    @pytest.mark.asyncio
    async def test_get_feedback_count_defaults_to_zero(self):
        class RedisStub:
            async def get(self, key):
                return None

        assert await get_feedback_count(RedisStub(), 9) == 0

    @pytest.mark.asyncio
    async def test_increment_feedback_count_updates_ttl(self):
        class RedisStub:
            def __init__(self):
                self.expire_calls = []

            async def incr(self, key):
                assert key == feedback_count_key(5)
                return 4

            async def expire(self, key, ttl):
                self.expire_calls.append((key, ttl))

        redis = RedisStub()
        count = await increment_feedback_count(redis, 5)

        assert count == 4
        assert redis.expire_calls == [(feedback_count_key(5), FEEDBACK_COUNT_TTL_SECONDS)]
