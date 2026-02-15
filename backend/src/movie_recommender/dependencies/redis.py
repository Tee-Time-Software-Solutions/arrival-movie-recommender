import redis
from fastapi import Depends

from movie_recommender.core.clients.redis import RedisClient
from movie_recommender.services.feed_manager.main import FeedManager
from movie_recommender.services.hydrator.main import MovieHydrator
from movie_recommender.services.recommender.main import Recommender


def get_sync_redis() -> redis.Redis:
    """Get synchronous Redis client."""
    return RedisClient().get_sync_client()


async def get_async_redis() -> redis.Redis:
    """Get asynchronous Redis client."""
    return await RedisClient().get_async_client()
