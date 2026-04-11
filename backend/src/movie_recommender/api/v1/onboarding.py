import logging
from typing import List

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import insert, update
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.core.settings.main import AppSettings
from movie_recommender.database.CRUD.movies import (
    get_movie_by_tmdb_id,
    get_onboarding_movie_cards,
    save_hydrated_movie,
)
from movie_recommender.database.CRUD.users import get_user_by_firebase_uid
from movie_recommender.database.models import movies, swipes, users
from movie_recommender.dependencies.database import get_db
from movie_recommender.dependencies.firebase import verify_user
from movie_recommender.dependencies.recommender import get_recommender
from movie_recommender.dependencies.redis import get_async_redis
from movie_recommender.schemas.requests.movies import MovieDetails
from movie_recommender.schemas.requests.onboarding import (
    OnboardingCompleteResponse,
    OnboardingMovieCard,
    OnboardingSearchResult,
    OnboardingSubmission,
)
from movie_recommender.services.hydrator.main import TMDBFetcher
from movie_recommender.services.onboarding.seed_movies import sample_onboarding_movies
from movie_recommender.services.recommender.main import Recommender
from movie_recommender.services.recommender.serving.user_vectors import (
    warm_start_vector,
)
from movie_recommender.services.recommender.serving.validation import (
    require_artifacts,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/onboarding")

_tmdb_fetcher: TMDBFetcher | None = None


def _get_tmdb_fetcher() -> TMDBFetcher:
    global _tmdb_fetcher
    if _tmdb_fetcher is None:
        _tmdb_fetcher = TMDBFetcher()
    return _tmdb_fetcher


@router.get("/movies")
async def get_onboarding_movies(
    db: AsyncSession = Depends(get_db),
    auth_user=Depends(verify_user()),
) -> List[OnboardingMovieCard]:
    """Return ~30 curated popular movies for the onboarding grid."""
    sampled = sample_onboarding_movies(per_genre=3)
    tmdb_ids = list({tmdb_id for _, tmdb_id in sampled})

    cards = await get_onboarding_movie_cards(db, tmdb_ids)

    if not cards:
        raise HTTPException(
            status_code=503,
            detail="Onboarding movies not yet seeded. Run the seed command first.",
        )

    return [OnboardingMovieCard(**card) for card in cards]


@router.get("/search")
async def search_onboarding_movies(
    query: str,
    auth_user=Depends(verify_user()),
) -> List[OnboardingSearchResult]:
    """Search TMDB for movies (step 2 of onboarding)."""
    if not query or len(query.strip()) < 2:
        return []

    settings = AppSettings()
    try:
        response = await _get_tmdb_fetcher()._client.get(
            f"{settings.tmdb.base_url}/search/movie",
            params={"api_key": settings.tmdb.api_key, "query": query.strip()},
        )
        data = response.json()
    except httpx.HTTPError:
        logger.warning("TMDB search failed", exc_info=True)
        raise HTTPException(status_code=502, detail="TMDB search unavailable")

    results = []
    for item in data.get("results", [])[:10]:
        poster_path = item.get("poster_path")
        release_date = item.get("release_date", "")
        results.append(
            OnboardingSearchResult(
                tmdb_id=item["id"],
                title=item.get("original_title") or item.get("title", ""),
                poster_url=f"{settings.tmdb.img_url}{poster_path}"
                if poster_path
                else None,
                release_year=int(release_date[:4]) if release_date else None,
            )
        )

    return results


async def _hydrate_tmdb_movie(db: AsyncSession, tmdb_id: int) -> int | None:
    """Fetch a movie by TMDB ID, insert into DB, return the DB movie ID."""
    settings = AppSettings()
    try:
        detail_res = (
            await _get_tmdb_fetcher()._client.get(
                f"{settings.tmdb.base_url}/movie/{tmdb_id}",
                params={
                    "api_key": settings.tmdb.api_key,
                    "append_to_response": "credits,videos,watch/providers,keywords",
                },
            )
        ).json()
    except httpx.HTTPError:
        logger.warning(f"Failed to fetch TMDB movie {tmdb_id}")
        return None

    if "id" not in detail_res:
        return None

    # Insert stub row to get a DB ID (caller is responsible for commit)
    result = await db.execute(
        insert(movies)
        .values(title=detail_res.get("original_title", "Unknown"))
        .returning(movies.c.id)
    )
    new_db_id = result.scalar_one()
    await db.flush()

    # Build full details and persist
    poster_path = detail_res.get("poster_path")
    release_date = detail_res.get("release_date", "")

    movie_details = MovieDetails(
        movie_db_id=new_db_id,
        tmdb_id=detail_res["id"],
        title=detail_res.get("original_title", ""),
        poster_url=f"{settings.tmdb.img_url}{poster_path}" if poster_path else "",
        release_year=int(release_date[:4]) if release_date else 0,
        rating=detail_res.get("vote_average", 0.0),
        genres=[g["name"] for g in detail_res.get("genres", [])],
        is_adult=detail_res.get("adult", False),
        synopsis=detail_res.get("overview", ""),
        runtime=detail_res.get("runtime", 0),
        trailer_url=_get_tmdb_fetcher()._extract_trailer_url(detail_res),
        cast=_get_tmdb_fetcher()._extract_cast_and_crew(detail_res),
        movie_providers=_get_tmdb_fetcher()._extract_providers(detail_res),
        keywords=_get_tmdb_fetcher()._extract_keywords(detail_res),
        collection=_get_tmdb_fetcher()._extract_collection(detail_res),
        production_companies=_get_tmdb_fetcher()._extract_production_companies(detail_res),
        genre_tmdb_ids=[g["id"] for g in detail_res.get("genres", [])],
    )

    await save_hydrated_movie(db, new_db_id, movie_details)
    return new_db_id


@router.post("/complete")
async def complete_onboarding(
    submission: OnboardingSubmission,
    db: AsyncSession = Depends(get_db),
    recommender: Recommender = Depends(get_recommender),
    redis_client=Depends(get_async_redis),
    auth_user=Depends(verify_user()),
) -> OnboardingCompleteResponse:
    """Submit onboarding selections and compute warm-start vector."""
    firebase_uid = auth_user["uid"]
    user = await get_user_by_firebase_uid(db, firebase_uid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.onboarding_completed:
        raise HTTPException(status_code=409, detail="Onboarding already completed")

    # Collect all unique DB movie IDs for vector computation
    all_movie_db_ids: set[int] = set(submission.grid_movie_ids)

    # Search picks: look up or hydrate from TMDB
    for tmdb_id in submission.search_movie_tmdb_ids:
        existing = await get_movie_by_tmdb_id(db, tmdb_id)
        if existing:
            all_movie_db_ids.add(existing.id)
        else:
            new_db_id = await _hydrate_tmdb_movie(db, tmdb_id)
            if new_db_id is not None:
                all_movie_db_ids.add(new_db_id)

    # Record all onboarding picks as "like" swipes (bulk insert)
    if all_movie_db_ids:
        await db.execute(
            insert(swipes).values(
                [
                    {
                        "user_id": user.id,
                        "movie_id": movie_id,
                        "action_type": "like",
                        "is_supercharged": False,
                    }
                    for movie_id in all_movie_db_ids
                ]
            )
        )

    # Mark onboarding as complete (same transaction as swipes)
    await db.execute(
        update(users)
        .where(users.c.id == user.id)
        .values(onboarding_completed=True)
    )

    # Single commit for all DB writes (swipes + hydrated movies + onboarding flag)
    await db.commit()

    # Mark as seen in Redis so they don't appear in the feed
    seen_key = f"seen:user:{user.id}"
    if redis_client and all_movie_db_ids:
        await redis_client.sadd(seen_key, *all_movie_db_ids)

    # Compute warm-start vector from selected movies
    artifacts = require_artifacts(
        recommender.artifacts, recommender._artifact_load_error
    )
    user_vector = warm_start_vector(artifacts, list(all_movie_db_ids))

    # Store in online vectors (same dict that current_user_vector checks first)
    # Note: despite type hints saying str, the existing pipeline uses int keys
    recommender.online_user_vectors[user.id] = user_vector

    movies_with_embeddings = sum(
        1 for mid in all_movie_db_ids if mid in artifacts.movie_id_to_index
    )

    return OnboardingCompleteResponse(
        onboarding_completed=True,
        movies_with_embeddings=movies_with_embeddings,
    )
