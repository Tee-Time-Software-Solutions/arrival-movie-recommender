"""Seed the onboarding movies into the DB by hydrating from TMDB.

Called automatically on app startup (fire-and-forget) if movies are missing.
Can also be run manually:
    python -m movie_recommender.services.onboarding.seed
"""

import asyncio
import logging

from sqlalchemy import func, select

from movie_recommender.core.settings.main import AppSettings
from movie_recommender.database.engine import DatabaseEngine
from movie_recommender.database.CRUD.movies import (
    get_movie_by_tmdb_id,
    save_hydrated_movie,
)
from movie_recommender.database.models import movies
from movie_recommender.schemas.requests.movies import MovieDetails
from movie_recommender.services.hydrator.main import TMDBFetcher
from movie_recommender.services.onboarding.seed_movies import ALL_SEED_TMDB_IDS

logger = logging.getLogger(__name__)


async def seed_onboarding_movies() -> None:
    """Check if onboarding movies exist in DB; hydrate any missing ones from TMDB.

    Most seed movies will already exist in the DB because they are popular
    MovieLens titles that get hydrated through normal feed usage. This function
    only fills in any gaps (e.g., on a fresh deployment).
    """
    engine = DatabaseEngine()
    settings = AppSettings()

    # Quick check: count how many seed movies are already hydrated
    async with engine.session_factory() as db:
        result = await db.execute(
            select(func.count())
            .select_from(movies)
            .where(
                movies.c.tmdb_id.in_(ALL_SEED_TMDB_IDS),
                movies.c.poster_url.isnot(None),
            )
        )
        existing_count = result.scalar_one()

    if existing_count >= len(ALL_SEED_TMDB_IDS):
        logger.info(
            f"All {len(ALL_SEED_TMDB_IDS)} onboarding movies already seeded, skipping."
        )
        return

    logger.info(
        f"Seeding onboarding movies: {existing_count}/{len(ALL_SEED_TMDB_IDS)} "
        f"already in DB, hydrating the rest..."
    )

    tmdb = TMDBFetcher()
    seeded = 0
    failed = 0

    for tmdb_id in sorted(ALL_SEED_TMDB_IDS):
        async with engine.session_factory() as db:
            existing = await get_movie_by_tmdb_id(db, tmdb_id)
            if existing and existing.poster_url:
                continue

        try:
            detail_res = (
                await tmdb._client.get(
                    f"{settings.tmdb.base_url}/movie/{tmdb_id}",
                    params={
                        "api_key": settings.tmdb.api_key,
                        "append_to_response": "credits,videos,watch/providers,keywords",
                    },
                )
            ).json()
        except Exception as e:
            logger.error(f"TMDB fetch failed for tmdb_id={tmdb_id}: {e}")
            failed += 1
            continue

        if "id" not in detail_res:
            logger.warning(f"No TMDB result for tmdb_id={tmdb_id}")
            failed += 1
            continue

        title = detail_res.get("original_title", "Unknown")
        poster_path = detail_res.get("poster_path")
        release_date = detail_res.get("release_date", "")

        async with engine.session_factory() as db:
            # If the movie exists as a stub (no poster), update it in place
            if existing:
                db_id = existing.id
            else:
                # Truly new movie — this is rare for seed movies
                from sqlalchemy import insert as sa_insert

                result = await db.execute(
                    sa_insert(movies).values(title=title).returning(movies.c.id)
                )
                db_id = result.scalar_one()
                await db.commit()

            movie_details = MovieDetails(
                movie_db_id=db_id,
                tmdb_id=detail_res["id"],
                title=title,
                poster_url=f"{settings.tmdb.img_url}{poster_path}"
                if poster_path
                else "",
                release_year=int(release_date[:4]) if release_date else 0,
                rating=detail_res.get("vote_average", 0.0),
                genres=[g["name"] for g in detail_res.get("genres", [])],
                is_adult=detail_res.get("adult", False),
                synopsis=detail_res.get("overview", ""),
                runtime=detail_res.get("runtime", 0),
                trailer_url=tmdb._extract_trailer_url(detail_res),
                cast=tmdb._extract_cast_and_crew(detail_res),
                movie_providers=tmdb._extract_providers(detail_res),
                keywords=tmdb._extract_keywords(detail_res),
                collection=tmdb._extract_collection(detail_res),
                production_companies=tmdb._extract_production_companies(detail_res),
                genre_tmdb_ids=[g["id"] for g in detail_res.get("genres", [])],
            )

            await save_hydrated_movie(db, db_id, movie_details)
            logger.info(f"Seeded: {title} (tmdb_id={tmdb_id}, db_id={db_id})")
            seeded += 1

    await tmdb._client.aclose()
    logger.info(f"Onboarding seed complete: seeded={seeded}, failed={failed}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(seed_onboarding_movies())
