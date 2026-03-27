from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.CRUD.interactions import create_swipe
from movie_recommender.database.CRUD.movies import get_movie_by_id
from movie_recommender.database.CRUD.users import get_user_by_firebase_uid
from movie_recommender.dependencies.database import get_db
from movie_recommender.dependencies.firebase import verify_user
from movie_recommender.dependencies.recommender import get_recommender
from movie_recommender.schemas.requests.interactions import (
    RegisteredFeedback,
    SwipeAction,
    SwipeRequest,
)
from movie_recommender.services.recommender.main import Recommender

router = APIRouter(prefix="/interactions")


@router.post(path="/{movie_id}/swipe")
async def register_movie_interaction(
    movie_id: int,
    swipe_data: SwipeRequest,
    db: AsyncSession = Depends(get_db),
    recommender: Recommender = Depends(get_recommender),
    auth_user=Depends(verify_user()),
) -> RegisteredFeedback:
    """
    1. Validate movie exists in DB
    2. Register swipe interaction for the user (using DB user id)
    3. Trigger recommender update
    """
    if swipe_data.action_type == SwipeAction.SKIP and swipe_data.is_supercharged:
        raise HTTPException(
            status_code=400, detail="Cannot have 'SKIP' interaction supercharged"
        )

    # 1) Database layer
    # TODO: validate movie_id exists in db, raise 404 if not
    # TODO: store interaction in db, get back interaction_id

    user_id = "demo2"  # TODO: replace with authenticated user ID from request context (e.g. auth dependency)
    interaction_id = str(movie_id)  # TODO: replace with actual interaction ID returned from DB after storing the swipe

    # 2) Provide info to the recommender
    recommender.set_user_feedback(
        user_id=user_db_id,
        movie_id=movie_id,
        interaction_type=swipe_data.action_type,
        is_supercharged=swipe_data.is_supercharged,
    )

    return RegisteredFeedback(
        interaction_id=db.interaction_id,
        movie_id=movie_id,
        action_type=swipe_data.action_type,
        is_supercharged=swipe_data.is_supercharged,
        registered=True,
    )
