import logging
from movie_recommender.core.settings.main import AppSettings
import redis
import asyncio
import json

from movie_recommender.schemas.movies import MovieDetails
from movie_recommender.schemas.users import UserPreferences
from movie_recommender.services.hydrator.main import MovieHydrator
from movie_recommender.services.recommender.main import Recommender

logger = logging.getLogger(__name__)


class FeedManager:
    """Manages ther redis queue and triggers background hydration of data (augmenting the data with TMDB)"""

    def __init__(
        self,
        recommender: Recommender,
        hydrator: MovieHydrator,
        redis_client: redis.Redis,
    ) -> None:
        self.recommender = recommender
        self.hydrator = hydrator
        self.redis_client = redis_client
        self.settings = AppSettings()

    async def get_next_movie(
        self, user_id: str, user_preferences: UserPreferences
    ) -> MovieDetails:
        """
        1. Extract from queue
        2. Refill queue if empty (blocking refill) or below threshold
        3. Hydrate movie info onto database
        """
        queue_key = f"feed:user:{user_id}"

        # 1) Pop next movie from Redis
        movie_data = await self.redis_client.lpop(queue_key)

        # 2) Check queue length
        queue_len = await self.redis_client.llen(queue_key)
        if queue_len < self.settings.app_logic.queue_min_capacity:
            asyncio.create_task(self.refill_queue(user_id, queue_key, user_preferences))

        if not movie_data:
            await self.refill_queue(user_id, queue_key, user_preferences)
            movie_data = await self.redis_client.lpop(queue_key)

        # Parse movie data
        logger.info(f"Movie data from Redis: {movie_data}")
        movie_id, movie_title = json.loads(movie_data)

        # 3) Get from DB or fetch and store
        return await self.hydrator.get_or_fetch_movie(
            movie_database_id=movie_id, movie_title=movie_title
        )

    async def refill_queue(
        self, user_id: str, queue_key: str, user_preferences: UserPreferences
    ):
        logger.info(f"Refilling queue for user {user_id}")
        # 1) Get recommendations (returns list of tuples)
        movies = self.recommender.get_top_n(
            user_id=user_id,
            n=self.settings.app_logic.batch_size,
            user_preferences=user_preferences,
        )
        logger.info(f"Got {len(movies)} recommendations from recommender")

        # 2) Ensure movies are in DB, then push to Redis
        for movie_id, movie_title in movies:
            await self.hydrator.get_or_fetch_movie(movie_id, movie_title)
            await self.redis_client.rpush(
                queue_key, json.dumps([movie_id, movie_title])
            )
