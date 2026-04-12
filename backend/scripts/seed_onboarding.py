"""Seed the onboarding movies into the DB by hydrating from TMDB."""

import asyncio
import logging
import sys

sys.path.insert(0, "src")

from movie_recommender.core.settings.main import AppSettings
from movie_recommender.database.engine import DatabaseEngine
from movie_recommender.database.CRUD.movies import get_movie_by_tmdb_id, save_hydrated_movie
from movie_recommender.services.hydrator.main import TMDBFetcher
from movie_recommender.services.onboarding.seed_movies import ALL_SEED_TMDB_IDS
from sqlalchemy import insert
from movie_recommender.database.models import movies

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    _ = AppSettings()
    engine = DatabaseEngine()
    tmdb = TMDBFetcher()
    settings = AppSettings()

    seeded = 0
    skipped = 0
    failed = 0

    for tmdb_id in sorted(ALL_SEED_TMDB_IDS):
        async with engine.session_factory() as db:
            existing = await get_movie_by_tmdb_id(db, tmdb_id)
            if existing and existing.tmdb_id:
                logger.info(f"Already in DB: {existing.title} (tmdb_id={tmdb_id})")
                skipped += 1
                continue

        # Fetch from TMDB by ID
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
            logger.error(f"TMDB fetch failed for {tmdb_id}: {e}")
            failed += 1
            continue

        if "id" not in detail_res:
            logger.warning(f"No TMDB result for {tmdb_id}")
            failed += 1
            continue

        title = detail_res.get("original_title", "Unknown")
        poster_path = detail_res.get("poster_path")
        release_date = detail_res.get("release_date", "")

        from movie_recommender.schemas.requests.movies import MovieDetails

        async with engine.session_factory() as db:
            # Create stub row
            result = await db.execute(
                insert(movies).values(title=title).returning(movies.c.id)
            )
            new_id = result.scalar_one()
            await db.commit()

            movie_details = MovieDetails(
                movie_db_id=new_id,
                tmdb_id=detail_res["id"],
                title=title,
                poster_url=f"{settings.tmdb.img_url}{poster_path}" if poster_path else "",
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

            await save_hydrated_movie(db, new_id, movie_details)
            logger.info(f"Seeded: {title} (tmdb_id={tmdb_id}, db_id={new_id})")
            seeded += 1

    await tmdb._client.aclose()
    logger.info(f"Done! Seeded={seeded}, Skipped={skipped}, Failed={failed}")


if __name__ == "__main__":
    asyncio.run(main())
