import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
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


async def _resolve_user_and_prefs(db: AsyncSession, firebase_uid: str):
    """Load internal user row and their preferences from the DB."""
    user = await get_user_by_firebase_uid(db, firebase_uid)
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
    return user, user_prefs


@router.get(path="/feed")
async def fetch_movies_feed(
    feed_manager: FeedManager = Depends(get_feed_manager),
    db: AsyncSession = Depends(get_db),
    auth_user=Depends(verify_user()),
) -> MovieDetails:
    """
    Pop next movie from redis queue, refill if needed, return hydrated movie details.
    """
    user, user_prefs = await _resolve_user_and_prefs(db, auth_user["uid"])

    movie = await feed_manager.get_next_movie(
        user_id=user.id, user_preferences=user_prefs
    )
    if not movie:
        raise HTTPException(status_code=404, detail="No movies found")
    return movie


@router.get(path="/feed/batch")
async def fetch_movies_feed_batch(
    count: int = Query(default=5, ge=1, le=20),
    feed_manager: FeedManager = Depends(get_feed_manager),
    db: AsyncSession = Depends(get_db),
    auth_user=Depends(verify_user()),
) -> List[MovieDetails]:
    """
    Return up to `count` movies in a single request.
    The first call may trigger a blocking refill (pre-hydrates & caches the batch).
    Subsequent pops are instant DB cache hits.
    """
    user, user_prefs = await _resolve_user_and_prefs(db, auth_user["uid"])

    movies: list[MovieDetails] = []
    for _ in range(count):
        movie = await feed_manager.get_next_movie(
            user_id=user.id, user_preferences=user_prefs
        )
        if movie:
            movies.append(movie)
        else:
            break

    if not movies:
        raise HTTPException(status_code=404, detail="No movies found")
    return movies


@router.delete(path="/feed")
async def flush_movies_feed(
    feed_manager: FeedManager = Depends(get_feed_manager),
    db: AsyncSession = Depends(get_db),
    auth_user=Depends(verify_user()),
):
    """Flush the user's Redis feed queue (e.g. after filter changes)."""
    user = await get_user_by_firebase_uid(db, auth_user["uid"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    await feed_manager.flush_feed(user.id)
    return {"flushed": True}
