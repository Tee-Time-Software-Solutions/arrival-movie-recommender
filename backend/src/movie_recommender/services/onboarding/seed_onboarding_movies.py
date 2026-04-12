"""Seed the onboarding movies into the DB by hydrating from TMDB.

Called automatically on app startup (fire-and-forget) if movies are missing.
Can also be run manually:
    python -m movie_recommender.services.onboarding.seed_onboarding_movies
"""

import asyncio
import logging

from movie_recommender.database.engine import DatabaseEngine
from movie_recommender.database.CRUD.movies import count_seeded_onboarding_movies
from movie_recommender.services.recommender.pipeline.hydrator.main import MovieHydrator
from movie_recommender.services.onboarding.seed_movies import (
    ALL_SEED_ML_IDS,
    SEED_MOVIES,
)

logger = logging.getLogger(__name__)


async def seed_onboarding_movies() -> None:
    """Hydrate any un-seeded onboarding movies from TMDB."""
    engine = DatabaseEngine()

    async with engine.session_factory() as db:
        existing_count = await count_seeded_onboarding_movies(db, list(ALL_SEED_ML_IDS))

    if existing_count >= len(ALL_SEED_ML_IDS):
        logger.info(f"All {len(ALL_SEED_ML_IDS)} onboarding movies already seeded.")
        return

    logger.info(
        f"Seeding: {existing_count}/{len(ALL_SEED_ML_IDS)} in DB, hydrating rest..."
    )

    hydrator = MovieHydrator(db_session_factory=engine.session_factory)
    seeded = 0
    failed = 0

    for entries in SEED_MOVIES.values():
        for ml_id, title in entries:
            details = await hydrator.get_or_fetch_movie(ml_id, title)
            if details:
                seeded += 1
            else:
                logger.warning(f"Failed to seed ml_id={ml_id} ({title!r})")
                failed += 1

    logger.info(f"Seed complete: seeded={seeded}, failed={failed}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(seed_onboarding_movies())
