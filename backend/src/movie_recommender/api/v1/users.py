from fastapi import APIRouter

from backend.src.movie_recommender.schemas.users import (
    UserPreferences,
    UserProfileSummary,
    UserStats,
)


router = APIRouter(prefix="/users")


@router.get(path="/me/summary")
async def get_full_profile_view() -> UserProfileSummary:
    """
    1. Fetch from db the user stats
    2. Fetch from db user profile info

     Dependencies:
        - Current user
    """


@router.update(path="/me/preferences")
async def update_preferences(updated_preferences: UserPreferences) -> UserPreferences:
    """
    1. Fetch from db the user stats

    Dependencies:
        - Current user
    """
