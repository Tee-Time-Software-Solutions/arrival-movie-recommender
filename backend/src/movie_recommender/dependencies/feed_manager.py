import redis
from fastapi import Depends

from movie_recommender.dependencies.recommender import get_recommender
from movie_recommender.dependencies.redis import get_async_redis
from movie_recommender.services.feed_manager.main import FeedManager
from movie_recommender.services.hydrator.main import MovieHydrator
from movie_recommender.services.recommender.main import Recommender


def get_feed_manager(
    redis_client: redis.Redis = Depends(get_async_redis),
    recommender: Recommender = Depends(get_recommender),
) -> FeedManager:
    """Create FeedManager with injected Redis client."""
    hydrator = MovieHydrator(db_session=None)  # TODO: inject when DB is ready
    return FeedManager(
        recommender=recommender, hydrator=hydrator, redis_client=redis_client
    )
