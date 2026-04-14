from neo4j import AsyncDriver

from movie_recommender.core.clients.neo4j import Neo4jClient


async def get_neo4j_driver() -> AsyncDriver:
    """Get async Neo4j driver."""
    return await Neo4jClient().get_async_driver()
