import asyncio
import logging
import random

from neo4j import AsyncDriver
from neo4j.exceptions import TransientError

logger = logging.getLogger(__name__)

CONSTRAINTS = [
    ("movie_tmdb_id", "Movie", "tmdb_id"),
    ("person_tmdb_id", "Person", "tmdb_id"),
    ("genre_tmdb_id", "Genre", "tmdb_id"),
    ("production_company_tmdb_id", "ProductionCompany", "tmdb_id"),
    ("collection_tmdb_id", "Collection", "tmdb_id"),
    ("keyword_tmdb_id", "Keyword", "tmdb_id"),
]

_MAX_RETRIES = 5


async def ensure_kg_schema(driver: AsyncDriver) -> None:
    """Create uniqueness constraints on all KG node types (idempotent).

    Retries on TransientError (deadlock) with jitter — gunicorn workers start
    simultaneously and can deadlock each other on concurrent constraint creation.
    """
    # Stagger workers so they don't all hit Neo4j at the exact same instant.
    await asyncio.sleep(random.uniform(0, 1.5))

    for constraint_name, label, prop in CONSTRAINTS:
        query = (
            f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
        )
        for attempt in range(_MAX_RETRIES):
            try:
                async with driver.session() as session:
                    await session.run(query)
                logger.info(f"Ensured constraint: {constraint_name} on :{label}.{prop}")
                break
            except TransientError:
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    raise

    logger.info("KG schema initialization complete")
