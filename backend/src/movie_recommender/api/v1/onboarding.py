import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.CRUD.interactions import create_swipes_bulk
from movie_recommender.database.CRUD.movies import (
    get_movie_by_id,
    get_movie_by_tmdb_id,
    get_onboarding_movie_cards,
)
from movie_recommender.database.CRUD.users import (
    get_user_by_firebase_uid,
    mark_onboarding_completed,
)
from movie_recommender.dependencies.database import get_db
from movie_recommender.dependencies.firebase import verify_user
from movie_recommender.dependencies.hydrator import get_movie_hydrator
from movie_recommender.dependencies.recommender import get_recommender
from movie_recommender.dependencies.redis import get_async_redis
from movie_recommender.schemas.requests.interactions import SwipeAction
from movie_recommender.schemas.requests.onboarding import (
    OnboardingCompleteResponse,
    OnboardingMovieCard,
    OnboardingSearchResult,
    OnboardingSubmission,
)
from movie_recommender.services.onboarding.seed_movies import sample_onboarding_movies
from movie_recommender.services.recommender.main import Recommender
from movie_recommender.services.recommender.pipeline.hydrator.main import MovieHydrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/onboarding")


@router.get("/movies")
async def get_onboarding_movies(
    db: AsyncSession = Depends(get_db),
    auth_user=Depends(verify_user()),
) -> List[OnboardingMovieCard]:
    """Return ~30 curated popular movies for the onboarding grid."""
    sampled = sample_onboarding_movies(per_genre=3)
    ml_ids = list({ml_id for _, ml_id, _ in sampled})

    cards = await get_onboarding_movie_cards(db, ml_ids)
    if not cards:
        raise HTTPException(
            status_code=503,
            detail="Onboarding movies not yet seeded. Run the seed command first.",
        )
    return [OnboardingMovieCard(**card) for card in cards]


@router.get("/search")
async def search_onboarding_movies(
    query: str,
    db: AsyncSession = Depends(get_db),
    hydrator: MovieHydrator = Depends(get_movie_hydrator),
    auth_user=Depends(verify_user()),
) -> List[OnboardingSearchResult]:
    """Search TMDB for movies during onboarding. Includes movie_db_id when already in DB."""
    if not query or len(query.strip()) < 2:
        return []

    try:
        results = await hydrator.tmdb.search_movies(query.strip())
    except Exception:
        logger.warning("TMDB search failed", exc_info=True)
        raise HTTPException(status_code=502, detail="TMDB search unavailable")

    output = []
    for item in results:
        db_movie = await get_movie_by_tmdb_id(db, item["id"])
        poster_path = item.get("poster_path")
        release_date = item.get("release_date", "")
        output.append(
            OnboardingSearchResult(
                movie_db_id=db_movie.id if db_movie else None,
                tmdb_id=item["id"],
                title=item.get("original_title") or item.get("title", ""),
                poster_url=f"{hydrator.tmdb.IMG_URL}{poster_path}"
                if poster_path
                else None,
                release_year=int(release_date[:4]) if release_date else None,
            )
        )
    return output


@router.post("/complete")
async def complete_onboarding(
    submission: OnboardingSubmission,
    db: AsyncSession = Depends(get_db),
    recommender: Recommender = Depends(get_recommender),
    redis_client=Depends(get_async_redis),
    hydrator: MovieHydrator = Depends(get_movie_hydrator),
    auth_user=Depends(verify_user()),
) -> OnboardingCompleteResponse:
    """Submit onboarding movie selections and initialise the user's recommendation vector."""
    user = await get_user_by_firebase_uid(db, auth_user["uid"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.onboarding_completed:
        raise HTTPException(status_code=409, detail="Onboarding already completed")

    movie_db_ids: list[int] = []
    for db_id in submission.movie_db_ids:
        movie = await get_movie_by_id(db, db_id)
        if not movie:
            continue
        details = await hydrator.get_or_fetch_movie(db_id, movie.title)
        if details:
            movie_db_ids.append(details.movie_db_id)

    await create_swipes_bulk(db, user.id, movie_db_ids, action_type="like")
    await mark_onboarding_completed(db, user.id)

    if redis_client and movie_db_ids:
        await redis_client.sadd(f"seen:user:{user.id}", *movie_db_ids)

    for mid in movie_db_ids:
        await recommender.set_user_feedback(
            user_id=user.id,
            movie_id=mid,
            interaction_type=SwipeAction.LIKE,
            is_supercharged=False,
        )

    return OnboardingCompleteResponse(onboarding_completed=True)
