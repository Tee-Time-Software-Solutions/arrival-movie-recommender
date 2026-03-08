from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

from movie_recommender.core.settings.main import AppSettings
import redis
import asyncio
import json

from movie_recommender.schemas.requests.movies import MovieDetails
from movie_recommender.services.hydrator.main import MovieHydrator
from movie_recommender.services.recommender.main import Recommender

if TYPE_CHECKING:
    from movie_recommender.schemas.requests.users import UserPreferences

logger = logging.getLogger(__name__)


# async def get_filtered_movies_for_user(db: AsyncSession, user_id: int) -> List[int]:
#     """Get list of movie IDs to exclude based on user preferences."""
#     from movie_recommender.database.CRUD.users import get_user_preferences
#     preferences = await get_user_preferences(db, user_id)
#     if not preferences:
#         return []
#     # TODO: query movies that don't match preference filters (genres, year range, rating)
#     return []


class FeedManager:
    """Manages the redis queue and triggers background hydration of data (augmenting the data with TMDB)"""

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
        self, user_id: int, user_preferences: UserPreferences | None = None
    ) -> MovieDetails:
        """
        1. Extract from queue
        2. Refill queue if empty (blocking refill) or below threshold
        3. Hydrate movie info onto database
        """
        queue_key = f"feed:user:{user_id}"

        movie_data = await self.redis_client.lpop(queue_key)

        queue_len = await self.redis_client.llen(queue_key)
        if queue_len < self.settings.app_logic.queue_min_capacity:
            asyncio.create_task(self.refill_queue(user_id, queue_key))

        if not movie_data:
            await self.refill_queue(user_id, queue_key)
            movie_data = await self.redis_client.lpop(queue_key)

        logger.info(f"Movie data from Redis: {movie_data}")
        movie_id, movie_title = json.loads(movie_data)

        return await self.hydrator.get_or_fetch_movie(
            movie_db_id=movie_id, movie_title=movie_title
        )

    async def refill_queue(self, user_id: int, queue_key: str):
        logger.info(f"Refilling queue for user {user_id}")

        # TODO: replace [] with get_filtered_movies_for_user(db, user_id) when preferences are implemented
        movies = self.recommender.get_top_n_recommendations(
            user_id=user_id,
            n=self.settings.app_logic.batch_size,
            list_of_filtered_movies=[],
        )
        logger.info(f"Got {len(movies)} recommendations from recommender")

        for movie_id, movie_title in movies:
            await self.hydrator.get_or_fetch_movie(movie_id, movie_title)
            await self.redis_client.rpush(
                queue_key, json.dumps([movie_id, movie_title])
            )
