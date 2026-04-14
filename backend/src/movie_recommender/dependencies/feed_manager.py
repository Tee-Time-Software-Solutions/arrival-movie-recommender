import redis
from fastapi import Depends

from movie_recommender.database.engine import DatabaseEngine
from movie_recommender.dependencies.recommender import get_recommender
from movie_recommender.dependencies.redis import get_async_redis
from movie_recommender.services.recommender.pipeline.feed_manager.main import (
    FeedManager,
)
from movie_recommender.services.recommender.pipeline.hydrator.main import MovieHydrator
from movie_recommender.services.recommender.main import Recommender


async def get_feed_manager(
    redis_client: redis.Redis = Depends(get_async_redis),
    recommender: Recommender = Depends(get_recommender),
) -> FeedManager:
    db_session_factory = DatabaseEngine().session_factory
    hydrator = MovieHydrator(db_session_factory=db_session_factory)
    return FeedManager(
        recommender=recommender,
        hydrator=hydrator,
        redis_client=redis_client,
        db_session_factory=db_session_factory,
    )
