import uuid

from fastapi import APIRouter, Depends

from movie_recommender.dependencies.recommender import get_recommender
from movie_recommender.schemas.interactions import (
    RegisteredFeedback,
    SwipeRequest,
)
from movie_recommender.services.recommender.main import Recommender

router = APIRouter(prefix="/interactions")


@router.post(path="/{movie_id}/swipe")
async def register_movie_feedback(
    movie_id: int,
    swipe_data: SwipeRequest,
    recommender: Recommender = Depends(get_recommender),
) -> RegisteredFeedback:
    """
    1. Register in db for a given movie like, dislike and if supercharged
    2. Trigger recommder update based on this info
    3. If movie not in db return 404 error

    Dependencies:
          - Extract current user
    """
    user_id = "1"  # TODO: replace with authenticated current user ID

    recommender.update_user(
        user_id=user_id,
        movie_id=movie_id,
        action_type=swipe_data.action_type,
        is_supercharged=swipe_data.is_supercharged,
    )

    return RegisteredFeedback(
        interaction_id=str(uuid.uuid4()),
        movie_id=movie_id,
        action_type=swipe_data.action_type,
        is_supercharged=swipe_data.is_supercharged,
        registered=True,
    )
