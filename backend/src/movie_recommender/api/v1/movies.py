import logging
from fastapi import APIRouter, Depends, HTTPException

from movie_recommender.dependencies.firebase import verify_user
from movie_recommender.schemas.requests.movies import MovieDetails
from movie_recommender.dependencies.feed_manager import get_feed_manager
from movie_recommender.services.feed_manager.main import FeedManager

router = APIRouter(prefix="/movies")

logger = logging.getLogger(__name__)


@router.get(path="/feed")
async def fetch_movies_feed(
    feed_manager: FeedManager = Depends(get_feed_manager),
    auth_user=Depends(verify_user()),
) -> MovieDetails:
    """
    Pop next movie from redis queue, refill if needed, return hydrated movie details.
    """
    logger.debug("Fetching next movie from feed")
    # TODO: resolve user_db_id from auth_user["uid"] and pass it here
    movie = await feed_manager.get_next_movie(user_id=1, user_preferences=None)
    logger.info(f"Got movie: {movie}")
    if not movie:
        raise HTTPException(status_code=404, detail="No movies found")
    return movie
