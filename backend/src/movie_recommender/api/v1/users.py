from fastapi import APIRouter, Depends

from movie_recommender.dependencies.firebase import verify_user
from movie_recommender.schemas.requests.users import (
    UserPreferences,
    UserProfileSummary,
)

router = APIRouter(prefix="/users")


@router.get(path="/{user_id}/summary")
async def get_full_profile_view(
    user_id: str, auth_user=Depends(verify_user(user_private_route=True))
) -> str:  # UserProfileSummary:
    """
    1. Fetch from db the user stats
    2. Fetch from db user profile info

     Dependencies:
        - Current user
    """
    return "All good"


@router.patch(path="/{user_id}/preferences")
async def update_preferences(
    user_id: str,
    updated_preferences: UserPreferences,
    auth_user=Depends(verify_user(user_private_route=True)),
) -> UserPreferences:
    """
    1. Fetch from db the user stats

    Dependencies:
        - Current user
    """
