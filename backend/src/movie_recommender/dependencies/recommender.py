from functools import lru_cache

from movie_recommender.services.recommender.main import Recommender


@lru_cache(maxsize=1)
def get_recommender() -> Recommender:
    return Recommender()


async def init_recommender_redis(redis_client) -> Recommender:
    """Initialize the singleton recommender with a Redis client."""
    recommender = get_recommender()
    recommender.set_redis(redis_client)
    return recommender
