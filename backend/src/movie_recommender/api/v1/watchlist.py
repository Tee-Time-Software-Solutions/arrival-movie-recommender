from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.CRUD.movies import (
    get_movie_by_id,
    movies_to_details_bulk,
)
from movie_recommender.database.CRUD.users import get_user_by_firebase_uid
from movie_recommender.database.CRUD.watchlist import (
    add_to_watchlist,
    get_user_watchlist,
    remove_from_watchlist,
)
from movie_recommender.dependencies.database import get_db
from movie_recommender.dependencies.firebase import verify_user
from movie_recommender.schemas.requests.movies import PaginatedMovieDetails
from movie_recommender.schemas.requests.watchlist import (
    WatchlistAddResponse,
    WatchlistRemoveResponse,
)

router = APIRouter(prefix="/watchlist")


@router.post(path="/{movie_id}")
async def add_movie_to_watchlist(
    movie_id: int,
    db: AsyncSession = Depends(get_db),
    auth_user=Depends(verify_user()),
) -> WatchlistAddResponse:
    user = await get_user_by_firebase_uid(db, auth_user["uid"])
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    movie = await get_movie_by_id(db, movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail="Movie not found")

    result = await add_to_watchlist(db, user.id, movie_id)
    if result is None:
        raise HTTPException(status_code=409, detail="Movie already in watchlist")

    return WatchlistAddResponse(movie_id=movie_id, added=True)


@router.delete(path="/{movie_id}")
async def remove_movie_from_watchlist(
    movie_id: int,
    db: AsyncSession = Depends(get_db),
    auth_user=Depends(verify_user()),
) -> WatchlistRemoveResponse:
    user = await get_user_by_firebase_uid(db, auth_user["uid"])
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    removed = await remove_from_watchlist(db, user.id, movie_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Movie not in watchlist")

    return WatchlistRemoveResponse(movie_id=movie_id, removed=True)


@router.get(path="")
async def get_watchlist_movies(
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    auth_user=Depends(verify_user()),
) -> PaginatedMovieDetails:
    user = await get_user_by_firebase_uid(db, auth_user["uid"])
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    movie_ids, total = await get_user_watchlist(db, user.id, limit, offset)
    items = await movies_to_details_bulk(db, movie_ids)

    return PaginatedMovieDetails(items=items, total=total, limit=limit, offset=offset)
