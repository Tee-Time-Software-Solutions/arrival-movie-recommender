import logging
import redis.asyncio as async_redis
import redis as sync_redis

from movie_recommender.core.settings import AppSettings

logger = logging.getLogger(__name__)


class RedisClient:
    """Singleton Redis client supporting both sync and async connections."""

    _instance = None
    _async_client = None
    _sync_client = None

    def __new__(cls, settings: AppSettings = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.settings = settings or AppSettings()
        return cls._instance

    async def get_async_client(self) -> async_redis.Redis:
        """Get async Redis client (for async operations)."""
        if not self._async_client:
            self._async_client = async_redis.from_url(
                url=self.settings.redis.url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=self.settings.redis.max_connections,
            )
            logger.debug(f"Created async Redis client: {id(self._async_client)}")
        return self._async_client

    def get_sync_client(self) -> sync_redis.Redis:
        """Get sync Redis client (for sync operations like RQ)."""
        if not self._sync_client:
            self._sync_client = sync_redis.from_url(
                url=self.settings.redis.url,
                decode_responses=False,
            )
            logger.debug(f"Created sync Redis client: {id(self._sync_client)}")
        return self._sync_client

    async def close(self) -> None:
        """Close all Redis connections."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
            logger.debug("Closed async Redis client")
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
            logger.debug("Closed sync Redis client")
