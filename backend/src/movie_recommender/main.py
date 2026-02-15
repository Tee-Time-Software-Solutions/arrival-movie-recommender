from contextlib import asynccontextmanager
import logging
from movie_recommender.core.logger.main import initialize_logger
from movie_recommender.core.settings.main import AppSettings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from movie_recommender.api.v1 import routers
from movie_recommender.core.clients.redis import RedisClient


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    initialize_logger()
    logger.info("Starting up application...")
    yield
    # Shutdown
    logger.info("Shutting down application...")
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
