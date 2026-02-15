from fastapi import APIRouter

from backend.src.movie_recommender.schemas.movies import MovieCard, MovieDetails

router = APIRouter(prefix="/movies")


@router.get(path="/feed")
async def fetch_movies_feed() -> MovieCard:
    """
    1. Pop from redis queue
    2. If length of rq is < threshold refill in a background task
          2.1 Get name of movie by calling recommender.get_next(preferences=None) # Preferneces will be added in the future
      If length == 0:
         Wait for the background task
    3. Returns popped movie

    To be expandend:
      - Add user preferences

    Dependencies:
      - Current user
    """


@router.get(path="/{movie_id}/details")
async def fetch_movie_details(movie_id: int) -> MovieDetails:
    """
    1. Fetch from cache/db the metadata of TMDB
    If it exists:
          return
    else:
          1. Sync request to TMDB, if its down return 404 error
          2. Save result to DB/Cache
          3. Return data to user
    """
