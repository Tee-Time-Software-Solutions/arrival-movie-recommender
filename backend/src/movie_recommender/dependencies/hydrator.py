from fastapi import Depends
from neo4j import AsyncDriver

from movie_recommender.database.engine import DatabaseEngine
from movie_recommender.dependencies.neo4j import get_neo4j_driver
from movie_recommender.services.recommender.pipeline.hydrator.main import MovieHydrator


async def get_movie_hydrator(
    neo4j_driver: AsyncDriver = Depends(get_neo4j_driver),
) -> MovieHydrator:
    return MovieHydrator(
        db_session_factory=DatabaseEngine().session_factory,
        neo4j_driver=neo4j_driver,
    )
