import asyncio
from contextlib import asynccontextmanager
import logging
from movie_recommender.core.logger.main import initialize_logger
from movie_recommender.core.settings.main import AppSettings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from movie_recommender.api.v1 import routers
from movie_recommender.core.clients.neo4j import Neo4jClient
from movie_recommender.core.clients.redis import RedisClient
from movie_recommender.core.clients.firebase import initialize_firebase
from movie_recommender.database.engine import DatabaseEngine
from movie_recommender.dependencies.recommender import init_recommender_redis
from movie_recommender.services.knowledge_graph.schema import ensure_kg_schema
from movie_recommender.services.onboarding.seed import seed_onboarding_movies
from movie_recommender.services.swipe_worker.main import drain_swipe_queue


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    initialize_logger()
    logger.info("Starting up application...")
    initialize_firebase(AppSettings())

    redis_client = await RedisClient().get_async_client()
    neo4j_driver = await Neo4jClient().get_async_driver()
    await ensure_kg_schema(neo4j_driver)
    await init_recommender_redis(redis_client)
    db_engine = DatabaseEngine()
    swipe_task = asyncio.create_task(
        drain_swipe_queue(redis_client, db_engine.session_factory)
    )

    # Seed onboarding movies if not already in DB (fire-and-forget)
    asyncio.create_task(seed_onboarding_movies())

    yield

    # Shutdown
    logger.info("Shutting down application...")
    swipe_task.cancel()
    try:
        await swipe_task
    except asyncio.CancelledError:
        pass
    await Neo4jClient().close()
    await RedisClient().close()


app = FastAPI(title="Movie Recommender", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

for router in routers:
    app.include_router(router, prefix="/api/v1")
