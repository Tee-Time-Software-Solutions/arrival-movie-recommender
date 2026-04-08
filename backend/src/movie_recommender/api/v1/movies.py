import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.CRUD.users import (
    get_user_by_firebase_uid,
    get_user_preferences,
    get_user_included_genres,
    get_user_excluded_genres,
)
from movie_recommender.dependencies.database import get_db
from movie_recommender.dependencies.firebase import verify_user
from movie_recommender.schemas.requests.movies import MovieDetails
from movie_recommender.schemas.requests.users import UserPreferences
from movie_recommender.dependencies.feed_manager import get_feed_manager
from movie_recommender.services.feed_manager.main import FeedManager

router = APIRouter(prefix="/movies")

logger = logging.getLogger(__name__)


@router.get(path="/feed")
async def fetch_movies_feed(
    feed_manager: FeedManager = Depends(get_feed_manager),
    db: AsyncSession = Depends(get_db),
    auth_user=Depends(verify_user()),
) -> MovieDetails:
    """
    Pop next movie from redis queue, refill if needed, return hydrated movie details.
    """
    logger.debug("Fetching next movie from feed")

    user = await get_user_by_firebase_uid(db, auth_user["uid"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    prefs_row = await get_user_preferences(db, user.id)
    inc = await get_user_included_genres(db, user.id)
    exc = await get_user_excluded_genres(db, user.id)

    user_prefs = UserPreferences(
        included_genres=inc,
        excluded_genres=exc,
        min_release_year=prefs_row.min_year if prefs_row else None,
        max_release_year=prefs_row.max_year if prefs_row else None,
        min_rating=prefs_row.min_rating if prefs_row else None,
        include_adult=prefs_row.include_adult
        if prefs_row and prefs_row.include_adult is not None
        else False,
    )

    movie = await feed_manager.get_next_movie(
        user_id=user.id, user_preferences=user_prefs
    )
    logger.info(f"Got movie: {movie}")
    if not movie:
        raise HTTPException(status_code=404, detail="No movies found")
    return movie
