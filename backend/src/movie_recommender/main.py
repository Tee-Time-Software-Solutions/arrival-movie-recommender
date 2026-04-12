import asyncio
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from movie_recommender.api.v1 import routers
from movie_recommender.core.clients.neo4j import Neo4jClient
from movie_recommender.core.clients.redis import RedisClient
from movie_recommender.core.clients.firebase import initialize_firebase
from movie_recommender.core.logger.main import initialize_logger
from movie_recommender.core.settings.main import AppSettings
from movie_recommender.dependencies.recommender import init_recommender_redis
from movie_recommender.services.knowledge_graph.schema import ensure_kg_schema
from movie_recommender.services.onboarding.seed_onboarding_movies import (
    seed_onboarding_movies,
)
from movie_recommender.services.recommender.pipeline.offline.models.als.main import (
    run_pipeline_cron_job,
)

from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)
scheduler = BackgroundScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    initialize_logger()
    logger.info("Starting up application...")
    initialize_firebase(AppSettings())

    redis_binary_client = await RedisClient().get_async_binary_client()
    neo4j_driver = await Neo4jClient().get_async_driver()
    await ensure_kg_schema(neo4j_driver)
    await init_recommender_redis(redis_binary_client)

    asyncio.create_task(seed_onboarding_movies())

    # Register cron job to rerun pipeline
    scheduler.add_job(run_pipeline_cron_job, "cron", hour=0, minute=0)
    scheduler.start()

    yield

    # Shutdown
    logger.info("Shutting down application...")
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
