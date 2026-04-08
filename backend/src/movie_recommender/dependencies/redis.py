import redis

from movie_recommender.core.clients.redis import RedisClient


def get_sync_redis() -> redis.Redis:
    """Get synchronous Redis client."""
    return RedisClient().get_sync_client()


async def get_async_redis() -> redis.Redis:
    """Get asynchronous Redis client."""
    return await RedisClient().get_async_client()
