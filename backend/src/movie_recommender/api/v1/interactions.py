import uuid

from fastapi import APIRouter, Depends, HTTPException

from movie_recommender.dependencies.rating_store import RatingStore, get_rating_store
from movie_recommender.dependencies.recommender import get_recommender
from movie_recommender.schemas.interactions import (
    RateRequest,
    RegisteredFeedback,
    SwipeAction,
    SwipeRequest,
)
from movie_recommender.services.hydrator.main import MovieHydrator
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
    if (swipe_data.action_type == SwipeAction.SKIP) and (
        swipe_data.is_supercharged
    ):
        raise HTTPException(
            status_code=400, detail="You cant have 'SKIP' interaction supercharged"
        )

    user_id = "demo2"  # TODO: replace with authenticated current user ID

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


@router.post(path="/{movie_id}/rate")
async def rate_movie(
    movie_id: int,
    rate_data: RateRequest,
    recommender: Recommender = Depends(get_recommender),
    rating_store: RatingStore = Depends(get_rating_store),
) -> dict:
    title = recommender.artifacts.movie_id_to_title.get(movie_id)
    if title is None:
        raise HTTPException(status_code=404, detail="Movie not found")

    hydrator = MovieHydrator(db_session=None)
    movie_details = await hydrator.get_or_fetch_movie(movie_id, title)
    if movie_details is None:
        raise HTTPException(status_code=404, detail="Could not hydrate movie from TMDB")

    user_id = "demo2"  # TODO: replace with authenticated current user ID
    rating_store.add_rating(user_id, str(movie_id), rate_data.rating, movie_details)

    return {"status": "ok", "movie_id": movie_id, "rating": rate_data.rating}
