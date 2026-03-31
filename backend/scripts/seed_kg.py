"""
Bulk seed the Neo4j knowledge graph from existing SQL movies.

Usage:
    python scripts/seed_kg.py [--limit N] [--offset N]

Reads all movies with a tmdb_id from PostgreSQL, fetches expanded TMDB metadata,
and upserts each into the KG. Rate-limited to respect TMDB API limits.
"""

import argparse
import asyncio
import logging
import time

from movie_recommender.core.settings.main import AppSettings
from movie_recommender.core.clients.neo4j import Neo4jClient
from movie_recommender.database.engine import DatabaseEngine
from movie_recommender.services.hydrator.main import TMDBFetcher
from movie_recommender.services.knowledge_graph.schema import ensure_kg_schema
from movie_recommender.services.knowledge_graph.writer import upsert_movie_to_kg

from sqlalchemy import select
from movie_recommender.database.models import movies

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# TMDB rate limit: ~40 requests per 10 seconds
RATE_LIMIT_DELAY = 0.25  # seconds between requests


async def seed_kg(limit: int | None = None, offset: int = 0) -> None:
    settings = AppSettings()
    db_engine = DatabaseEngine()
    neo4j_client = Neo4jClient(settings)
    neo4j_driver = await neo4j_client.get_async_driver()
    tmdb_fetcher = TMDBFetcher()

    await ensure_kg_schema(neo4j_driver)

    async with db_engine.session_factory() as db:
        query = (
            select(movies.c.id, movies.c.tmdb_id, movies.c.title)
            .where(movies.c.tmdb_id.isnot(None))
            .order_by(movies.c.id)
            .offset(offset)
        )
        if limit:
            query = query.limit(limit)

        result = await db.execute(query)
        movie_rows = result.fetchall()

    total = len(movie_rows)
    logger.info(f"Found {total} movies to seed into KG (offset={offset})")

    success = 0
    failed = 0
    for i, row in enumerate(movie_rows):
        try:
            details = tmdb_fetcher._fetch_tmdb_metadata(row.id, row.title)
            if details:
                await upsert_movie_to_kg(neo4j_driver, details)
                success += 1
            else:
                logger.warning(f"No TMDB data for: {row.title}")
                failed += 1
        except Exception:
            logger.error(f"Failed to seed: {row.title}", exc_info=True)
            failed += 1

        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i + 1}/{total} (success={success}, failed={failed})")

        time.sleep(RATE_LIMIT_DELAY)

    logger.info(
        f"Seeding complete: {success} succeeded, {failed} failed out of {total}"
    )
    await neo4j_client.close()


def main():
    parser = argparse.ArgumentParser(description="Seed Neo4j KG from SQL movies")
    parser.add_argument("--limit", type=int, default=None, help="Max movies to process")
    parser.add_argument("--offset", type=int, default=0, help="Offset to start from")
    args = parser.parse_args()

    asyncio.run(seed_kg(limit=args.limit, offset=args.offset))


if __name__ == "__main__":
    main()
