import math

from movie_recommender.schemas.requests.interactions import SwipeAction

FEEDBACK_COUNT_KEY_PREFIX = "feedback_count:user:"
FEEDBACK_COUNT_TTL_SECONDS = 30 * 24 * 60 * 60


def feedback_count_key(user_id: int) -> str:
    return f"{FEEDBACK_COUNT_KEY_PREFIX}{user_id}"


def adaptive_learning_rate(
    base_learning_rate: float,
    feedback_count: int,
    strength: float,
) -> float:
    """Boost early updates, then decay smoothly back toward the base rate."""
    if strength <= 0:
        return base_learning_rate

    count = max(feedback_count, 0)
    multiplier = 1.0 + (strength / math.sqrt(1.0 + count))
    return base_learning_rate * multiplier


def counts_for_adaptation(interaction_type: SwipeAction) -> bool:
    return interaction_type in {SwipeAction.LIKE, SwipeAction.DISLIKE}


async def get_feedback_count(redis_client, user_id: int) -> int:
    raw_count = await redis_client.get(feedback_count_key(user_id))
    if raw_count is None:
        return 0
    return int(raw_count)


async def increment_feedback_count(
    redis_client,
    user_id: int,
    ttl_seconds: int = FEEDBACK_COUNT_TTL_SECONDS,
) -> int:
    key = feedback_count_key(user_id)
    count = await redis_client.incr(key)
    await redis_client.expire(key, ttl_seconds)
    return int(count)
