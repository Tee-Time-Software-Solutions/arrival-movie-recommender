import logging

from movie_recommender.core.settings.main import AppSettings
import redis
import asyncio
import json

from movie_recommender.schemas.requests.movies import MovieDetails
from movie_recommender.services.hydrator.main import MovieHydrator
from movie_recommender.services.recommender.main import Recommender

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
        neo4j_driver=None,
        db_session_factory=None,
    ) -> None:
        self.recommender = recommender
        self.hydrator = hydrator
        self.redis_client = redis_client
        self.neo4j_driver = neo4j_driver
        self.db_session_factory = db_session_factory
        self.settings = AppSettings()

    async def get_next_movie(
        self, user_id: int, user_preferences: None = None
    ) -> MovieDetails:
        """
        1. Extract from queue
        2. Refill queue if empty (blocking refill) or below threshold
        3. Hydrate movie info onto database
        """
        queue_key = f"feed:user:{user_id}"

        movie_data = await self.redis_client.lpop(queue_key)

        if not movie_data:
            await self.refill_queue(user_id, queue_key)
            movie_data = await self.redis_client.lpop(queue_key)
        else:
            queue_len = await self.redis_client.llen(queue_key)
            if queue_len < self.settings.app_logic.queue_min_capacity:
                asyncio.create_task(self.refill_queue(user_id, queue_key))

        logger.info(f"Movie data from Redis: {movie_data}")
        movie_id, movie_title = json.loads(movie_data)

        movie_details = await self.hydrator.get_or_fetch_movie(
            movie_db_id=movie_id, movie_title=movie_title
        )

        if movie_details and movie_details.tmdb_id and self.neo4j_driver:
            movie_details = await self._attach_explanation(user_id, movie_details)

        return movie_details

    async def _attach_explanation(
        self, user_id: int, movie_details: MovieDetails
    ) -> MovieDetails:
        """Attach KG explanation to movie details. Never blocks or raises."""
        try:
            from movie_recommender.services.knowledge_graph.explainer import (
                explain_recommendation,
            )
            from movie_recommender.schemas.requests.movies import (
                EntityReference,
                ExplanationResponse,
            )

            result = await asyncio.wait_for(
                explain_recommendation(
                    neo4j_driver=self.neo4j_driver,
                    redis_client=self.redis_client,
                    db_session_factory=self.db_session_factory,
                    user_id=user_id,
                    movie_tmdb_id=movie_details.tmdb_id,
                ),
                timeout=0.1,  # 100ms budget
            )
            if result:
                movie_details.explanation = ExplanationResponse(
                    text=result.text,
                    entities=[
                        EntityReference(
                            entity_type=e.entity_type,
                            tmdb_id=e.tmdb_id,
                            name=e.name,
                        )
                        for e in result.entities
                    ],
                    confidence=result.confidence,
                )
        except asyncio.TimeoutError:
            logger.debug("Explanation timed out, serving without")
        except Exception:
            logger.warning("Explanation generation failed", exc_info=True)
        return movie_details

    async def refill_queue(self, user_id: int, queue_key: str):
        logger.info(f"Refilling queue for user {user_id}")

        # Flush stale entries so the user gets fresh recommendations
        await self.redis_client.delete(queue_key)

        ranked_movie_ids = await self.recommender.get_top_n_recommendations(
            user_id=user_id,
            list_of_movie_ids=list(self.recommender.artifacts.movie_id_to_index.keys()),
        )
        movies = [
            (
                movie_id,
                self.recommender.artifacts.movie_id_to_title.get(
                    movie_id, f"movie_{movie_id}"
                ),
            )
            for movie_id in ranked_movie_ids[: self.settings.app_logic.batch_size]
        ]
        logger.info(
            "Got %s recommendations from recommender",
            len(movies),
        )

        hydrated = await asyncio.gather(
            *(self.hydrator.get_or_fetch_movie(mid, title) for mid, title in movies)
        )
        for (movie_id, movie_title), result in zip(movies, hydrated):
            if result is not None:
                await self.redis_client.rpush(
                    queue_key, json.dumps([movie_id, movie_title])
                )
