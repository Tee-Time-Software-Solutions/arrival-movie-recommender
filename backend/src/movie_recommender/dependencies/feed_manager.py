import redis
from fastapi import Depends
from neo4j import AsyncDriver

from movie_recommender.database.engine import DatabaseEngine
from movie_recommender.dependencies.neo4j import get_neo4j_driver
from movie_recommender.dependencies.recommender import get_recommender
from movie_recommender.dependencies.redis import get_async_redis
from movie_recommender.services.feed_manager.main import FeedManager
from movie_recommender.services.hydrator.main import MovieHydrator
from movie_recommender.services.recommender.main import Recommender


async def get_feed_manager(
    redis_client: redis.Redis = Depends(get_async_redis),
    recommender: Recommender = Depends(get_recommender),
    neo4j_driver: AsyncDriver = Depends(get_neo4j_driver),
) -> FeedManager:
    db_session_factory = DatabaseEngine().session_factory
    hydrator = MovieHydrator(
        db_session_factory=db_session_factory, neo4j_driver=neo4j_driver
    )
    return FeedManager(
        recommender=recommender,
        hydrator=hydrator,
        redis_client=redis_client,
        neo4j_driver=neo4j_driver,
        db_session_factory=db_session_factory,
    )
