from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.CRUD.users import (
    create_user,
    get_user_by_firebase_uid,
    get_user_analytics,
    get_user_preferences,
    get_user_included_genres,
    get_user_excluded_genres,
    update_user_preferences,
)
from movie_recommender.database.CRUD.interactions import get_user_liked_movies
from movie_recommender.database.CRUD.movies import movie_to_details
from movie_recommender.dependencies.database import get_db
from movie_recommender.dependencies.firebase import verify_user
from movie_recommender.schemas.requests.movies import PaginatedMovieDetails
from movie_recommender.schemas.requests.users import (
    UserAnalytics,
    UserCreatedResponse,
    UserDisplayInfo,
    UserPreferences,
    UserProfileSummary,
    UserCreate,
)

router = APIRouter(prefix="/users")


@router.get(path="/{user_id}/summary")
async def get_full_profile_view(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    auth_user=Depends(verify_user(user_private_route=True)),
) -> UserProfileSummary:
    """Fetch user profile info and usage stats."""
    user = await get_user_by_firebase_uid(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    analytics = await get_user_analytics(db, user.id)
    prefs = await get_user_preferences(db, user.id)
    inc = await get_user_included_genres(db, user.id)
    exc = await get_user_excluded_genres(db, user.id)

    return UserProfileSummary(
        profile=UserDisplayInfo(
            username=user.email.split("@")[0],
            avatar_url=user.profile_image_url or "https://placeholder.com/avatar.png",
            joined_at=str(user.created_at),
        ),
        stats=UserAnalytics(**analytics),
        preferences=UserPreferences(
            included_genres=inc,
            excluded_genres=exc,
            min_release_year=prefs.min_year if prefs else None,
            max_release_year=prefs.max_year if prefs else None,
            min_rating=prefs.min_rating if prefs else None,
            include_adult=prefs.include_adult
            if prefs and prefs.include_adult is not None
            else False,
        ),
    )


@router.get(path="/{user_id}/liked-movies")
async def get_liked_movies(
    user_id: str,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    auth_user=Depends(verify_user(user_private_route=True)),
) -> PaginatedMovieDetails:
    """Return the user's liked movies, most recent first."""
    user = await get_user_by_firebase_uid(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    movie_ids, total = await get_user_liked_movies(db, user.id, limit, offset)
    items = [await movie_to_details(db, mid) for mid in movie_ids]

    return PaginatedMovieDetails(items=items, total=total, limit=limit, offset=offset)


@router.patch(path="/{user_id}/preferences")
async def update_preferences(
    user_id: str,
    updated_preferences: UserPreferences,
    db: AsyncSession = Depends(get_db),
    auth_user=Depends(verify_user(user_private_route=True)),
) -> UserPreferences:
    """Update user preferences (year range, rating threshold, adult content, genres)."""
    user = await get_user_by_firebase_uid(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    await update_user_preferences(
        db,
        user_id=user.id,
        min_year=updated_preferences.min_release_year,
        max_year=updated_preferences.max_release_year,
        min_rating=updated_preferences.min_rating,
        include_adult=updated_preferences.include_adult,
        included_genre_names=updated_preferences.included_genres,
        excluded_genre_names=updated_preferences.excluded_genres,
    )
    return updated_preferences


@router.post(path="/register")
async def register_new_user(
    user_info: UserCreate,
    db: AsyncSession = Depends(get_db),
    auth_user=Depends(verify_user()),
) -> UserCreatedResponse:
    """Register a new user in the database."""
    existing = await get_user_by_firebase_uid(db, user_info.firebase_uid)
    if existing:
        raise HTTPException(status_code=409, detail="User already registered")

    user = await create_user(db, user_info)
    return UserCreatedResponse(
        id=user.id,
        firebase_uid=user.firebase_uid,
        profile_image_url=user.profile_image_url or "",
        email=user.email,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )
