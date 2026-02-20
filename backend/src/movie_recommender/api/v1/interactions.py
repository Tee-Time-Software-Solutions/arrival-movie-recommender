from fastapi import APIRouter, Depends, HTTPException

from movie_recommender.dependencies.feed_manager import get_feed_manager
from movie_recommender.dependencies.recommender import get_recommender
from movie_recommender.schemas.interactions import (
    RegisteredFeedback,
    SwipeAction,
    SwipeRequest,
)
from movie_recommender.services.feed_manager.main import FeedManager
from movie_recommender.services.recommender.main import Recommender

router = APIRouter(prefix="/interactions")


@router.post(path="/{movie_id}/swipe")
async def register_movie_interaction(
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

    # 0) Validate input
    if (swipe_data.action_type == SwipeAction.SKIP.value) and (
        swipe_data.is_supercharged
    ):
        return HTTPException(
            status_code=400, detail="You cant have 'SKIP' interaction supercharged"
        )

    # 1) Database layer
    # TODO: validate movie_id exists in db, raise 404 if not
    # TODO: store interaction in db, get back interaction_id
    # TODO: extract user_id

    # 2) Provide info to the recommender
    recommender.set_user_feedback(
        user_id=user_id,
        movie_id=movie_id,
        interaction_type=swipe_data.action_type,
        is_supercharged=swipe_data.is_supercharged,
    )

    return RegisteredFeedback(
        interaction_id=db.interaction_id,
        movie_id=movie_id,
        action_type=swipe_data.action_type,
        is_supercharged=swipe_data.is_supercharged,
        registered=error.detected,
    )
