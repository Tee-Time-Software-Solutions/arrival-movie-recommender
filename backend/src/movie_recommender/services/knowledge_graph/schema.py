import logging

from neo4j import AsyncDriver

logger = logging.getLogger(__name__)

CONSTRAINTS = [
    ("movie_tmdb_id", "Movie", "tmdb_id"),
    ("person_tmdb_id", "Person", "tmdb_id"),
    ("genre_tmdb_id", "Genre", "tmdb_id"),
    ("production_company_tmdb_id", "ProductionCompany", "tmdb_id"),
    ("collection_tmdb_id", "Collection", "tmdb_id"),
    ("keyword_tmdb_id", "Keyword", "tmdb_id"),
]


async def ensure_kg_schema(driver: AsyncDriver) -> None:
    """Create uniqueness constraints on all KG node types (idempotent)."""
    async with driver.session() as session:
        for constraint_name, label, prop in CONSTRAINTS:
            query = (
                f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
            )
            await session.run(query)
            logger.info(f"Ensured constraint: {constraint_name} on :{label}.{prop}")

    logger.info("KG schema initialization complete")
