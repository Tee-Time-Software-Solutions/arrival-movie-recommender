import logging

from movie_recommender.core.settings.main import AppSettings
import redis
import asyncio
import json

from movie_recommender.database.CRUD.movies import get_filtered_movie_ids
from movie_recommender.schemas.requests.movies import MovieDetails
from movie_recommender.schemas.requests.users import UserPreferences
from movie_recommender.services.hydrator.main import MovieHydrator
from movie_recommender.services.recommender.main import Recommender

logger = logging.getLogger(__name__)

SEEN_KEY_PREFIX = "seen:user:"


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
        self, user_id: int, user_preferences: UserPreferences | None = None
    ) -> MovieDetails:
        """
        1. Extract from queue
        2. Refill queue if empty (blocking refill) or below threshold
        3. Skip any movie the user already interacted with (Redis seen set)
        4. Hydrate movie info onto database
        """
        queue_key = f"feed:user:{user_id}"

        # Redis seen set is updated synchronously on every swipe,
        # so it's always current for the active session.
        seen_ids: set[int] = set()
        redis_seen = await self.redis_client.smembers(f"{SEEN_KEY_PREFIX}{user_id}")
        if redis_seen:
            seen_ids = {int(m) for m in redis_seen}

        movie_data = await self._pop_unseen(queue_key, seen_ids)

        if not movie_data:
            await self.refill_queue(user_id, queue_key, user_preferences)
            movie_data = await self._pop_unseen(queue_key, seen_ids)
        else:
            queue_len = await self.redis_client.llen(queue_key)
            if queue_len < self.settings.app_logic.queue_min_capacity:
                asyncio.create_task(
                    self.refill_queue(user_id, queue_key, user_preferences)
                )

        if not movie_data:
            logger.warning("No movies available for user %s after refill", user_id)
            return None

        logger.info(f"Movie data from Redis: {movie_data}")
        movie_id, movie_title = json.loads(movie_data)

        # Mark as seen immediately so concurrent/future requests won't serve it again
        await self.redis_client.sadd(f"{SEEN_KEY_PREFIX}{user_id}", movie_id)

        movie_details = await self.hydrator.get_or_fetch_movie(
            movie_db_id=movie_id, movie_title=movie_title
        )

        if movie_details and movie_details.tmdb_id and self.neo4j_driver:
            movie_details = await self._attach_explanation(user_id, movie_details)

        return movie_details

    async def _pop_unseen(self, queue_key: str, seen_ids: set[int]) -> str | None:
        """Pop entries from the queue, skipping any the user already interacted with."""
        while True:
            raw = await self.redis_client.lpop(queue_key)
            if raw is None:
                return None
            movie_id, _ = json.loads(raw)
            if int(movie_id) not in seen_ids:
                return raw

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

    async def refill_queue(
        self,
        user_id: int,
        queue_key: str,
        user_preferences: UserPreferences | None = None,
    ):
        logger.info(f"Refilling queue for user {user_id}")

        # Flush stale entries so the user gets fresh recommendations
        await self.redis_client.delete(queue_key)

        # Invalidate the recommender's cached seen set so it reloads from Redis
        self.recommender.user_seen_movie_ids.pop(user_id, None)

        # Determine candidate movie IDs — if filters are active, narrow down
        # to movies already in the DB that match genre/year criteria.
        # This avoids TMDB calls entirely since filtered movies are pre-cached.
        all_movie_ids = list(self.recommender.artifacts.movie_id_to_index.keys())
        has_filters = user_preferences and (
            user_preferences.included_genres
            or user_preferences.min_release_year is not None
            or user_preferences.max_release_year is not None
        )

        if has_filters and self.db_session_factory:
            async with self.db_session_factory() as db:
                matching_ids = await get_filtered_movie_ids(
                    db,
                    genre_names=user_preferences.included_genres or None,
                    min_year=user_preferences.min_release_year,
                    max_year=user_preferences.max_release_year,
                )
            # Intersect with recommender's known movies
            candidate_ids = [mid for mid in all_movie_ids if mid in matching_ids]
            logger.info(
                "Filtered candidates: %d/%d movies match preferences",
                len(candidate_ids),
                len(all_movie_ids),
            )
        else:
            candidate_ids = all_movie_ids

        ranked_movie_ids = await self.recommender.get_top_n_recommendations(
            user_id=user_id,
            list_of_movie_ids=candidate_ids,
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
