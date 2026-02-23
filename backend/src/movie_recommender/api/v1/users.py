import logging
from typing import List

from fastapi import APIRouter, Depends

from movie_recommender.dependencies.rating_store import RatingStore, get_rating_store
from movie_recommender.dependencies.recommender import get_recommender
from movie_recommender.schemas.interactions import RatedMovie
from movie_recommender.schemas.movies import MovieDetails
from movie_recommender.schemas.users import (
    UserPreferences,
    UserProfileSummary,
)
from movie_recommender.services.hydrator.main import MovieHydrator
from movie_recommender.services.recommender.main import Recommender


router = APIRouter(prefix="/users")

logger = logging.getLogger(__name__)


@router.get(path="/me/recommendations")
async def get_recommendations(
    recommender: Recommender = Depends(get_recommender),
) -> List[MovieDetails]:
    """
    Return the top-N ML recommendations for the current user,
    hydrated with TMDB metadata (poster, genres, synopsis, etc.).
    """
    user_id = "demo2"  # TODO: replace with authenticated current user ID
    raw_recs = recommender.get_top_n(user_id, n=5, user_preferences=None)

    hydrator = MovieHydrator(db_session=None)
    results: List[MovieDetails] = []
    for movie_id, movie_title in raw_recs:
        movie = await hydrator.get_or_fetch_movie(movie_id, movie_title)
        if movie is not None:
            results.append(movie)

    return results


@router.get(path="/me/top-rated")
async def get_top_rated(
    rating_store: RatingStore = Depends(get_rating_store),
) -> List[RatedMovie]:
    user_id = "demo2"  # TODO: replace with authenticated current user ID
    return rating_store.get_top_rated(user_id)


@router.get(path="/me/summary")
async def get_full_profile_view() -> UserProfileSummary:
    """
    1. Fetch from db the user stats
    2. Fetch from db user profile info

     Dependencies:
        - Current user
    """


@router.patch(path="/me/preferences")
async def update_preferences(updated_preferences: UserPreferences) -> UserPreferences:
    """
    1. Fetch from db the user stats

    Dependencies:
        - Current user
    """
