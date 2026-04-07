import logging

from neo4j import AsyncGraphDatabase, AsyncDriver

from movie_recommender.core.settings import AppSettings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Singleton Neo4j client using the async driver."""

    _instance = None
    _driver: AsyncDriver | None = None

    def __new__(cls, settings: AppSettings = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.settings = settings or AppSettings()
        return cls._instance

    async def get_async_driver(self) -> AsyncDriver:
        """Get async Neo4j driver (created lazily on first call)."""
        if not self._driver:
            neo4j_settings = self.settings.neo4j
            self._driver = AsyncGraphDatabase.driver(
                neo4j_settings.uri,
                auth=(neo4j_settings.username, neo4j_settings.password),
            )
            logger.debug(f"Created async Neo4j driver: {id(self._driver)}")
        return self._driver

    async def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.debug("Closed async Neo4j driver")
