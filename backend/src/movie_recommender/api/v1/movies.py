import logging
from fastapi import APIRouter, Depends, HTTPException

from movie_recommender.schemas.movies import MovieDetails
from movie_recommender.dependencies.feed_manager import get_feed_manager
from movie_recommender.services.feed_manager.main import FeedManager

router = APIRouter(prefix="/movies")

logger = logging.getLogger(__name__)


@router.get(path="/feed")
async def fetch_movies_feed(
    feed_manager: FeedManager = Depends(get_feed_manager),
) -> MovieDetails:
    """
    1. Pop from redis queue
    2. If length  < threshold or == 0:
            - Refill in a background task (bg job if rq.length > 0)
                  - Augment data with metadata
    3. Get name of movie by calling recommender.get_next(preferences=None) # Preferneces will be added in the future
    3. Returns popped movie

    To be expandend:
      - Add user preferences

    Dependencies:
      - Current user
    """
    logger.debug("Fetching next movie from feed")
    movie = await feed_manager.get_next_movie(user_id="demo2", user_preferences=None)
    logger.info(f"Got movie: {movie}")
    if not movie:
        raise HTTPException(status_code=404, detail="No movies found")
    return movie
