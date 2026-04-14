from movie_recommender.database.engine import DatabaseEngine
from movie_recommender.services.recommender.pipeline.hydrator.main import MovieHydrator


async def get_movie_hydrator() -> MovieHydrator:
    return MovieHydrator(db_session_factory=DatabaseEngine().session_factory)
