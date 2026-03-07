from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.CRUD.users import (
    create_user,
    get_user_by_firebase_uid,
    update_user_preferences,
)
from movie_recommender.dependencies.database import get_db
from movie_recommender.dependencies.firebase import verify_user
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

    return UserProfileSummary(
        profile=UserDisplayInfo(
            username=user.email.split("@")[0],
            avatar_url=user.profile_image_url or "https://placeholder.com/avatar.png",
            joined_at=str(user.created_at),
        ),
        stats=UserAnalytics(
            total_swipes=0,
            total_likes=0,
            total_dislikes=0,
            top_genres=[],
        ),
        preferences=UserPreferences(
            preferred_genres=[],
            min_release_year=1900,
            include_adult=False,
            movie_providers=[],
        ),
    )


@router.patch(path="/{user_id}/preferences")
async def update_preferences(
    user_id: str,
    updated_preferences: UserPreferences,
    db: AsyncSession = Depends(get_db),
    auth_user=Depends(verify_user(user_private_route=True)),
) -> UserPreferences:
    """Update user preferences (year range, rating threshold, genres)."""
    user = await get_user_by_firebase_uid(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    await update_user_preferences(
        db,
        user_id=user.id,
        min_year=updated_preferences.min_release_year,
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
