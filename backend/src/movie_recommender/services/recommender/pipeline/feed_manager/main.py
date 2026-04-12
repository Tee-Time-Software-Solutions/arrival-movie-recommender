import asyncio
import json
import logging
from typing import Optional

import redis

from movie_recommender.core.settings.main import AppSettings
from movie_recommender.schemas.requests.movies import MovieDetails
from movie_recommender.schemas.requests.users import UserPreferences
from movie_recommender.services.recommender.pipeline.hydrator.main import MovieHydrator
from movie_recommender.services.recommender.main import Recommender

logger = logging.getLogger(__name__)

SEEN_KEY_PREFIX = "seen:user:"


class FeedManager:
    """Manages the Redis queue and triggers background hydration of data."""

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

    async def flush_feed(self, user_id: int) -> None:
        """Delete the user's Redis feed queue (e.g. after filter changes)."""
        queue_key = f"feed:user:{user_id}"
        await self.redis_client.delete(queue_key)

    async def get_next_movie(
        self,
        user_id: int,
        user_preferences: Optional[UserPreferences] = None,
    ) -> MovieDetails:
        """
        1. Pop next movie from the queue
        2. Refill queue if empty (blocking) or below threshold (background)
        3. Mark as seen in Redis
        4. Hydrate and return movie details
        """
        queue_key = f"feed:user:{user_id}"

        movie_data = await self.redis_client.lpop(queue_key)

        if not movie_data:
            await self.refill_queue(user_id, queue_key, user_preferences)
            movie_data = await self.redis_client.lpop(queue_key)
        else:
            queue_len = await self.redis_client.llen(queue_key)
            if queue_len < self.settings.app_logic.queue_min_capacity:
                asyncio.create_task(
                    self.refill_queue(user_id, queue_key, user_preferences)
                )

        if not movie_data:
            logger.warning("No movies available for user %s after refill", user_id)
            return None

        movie_id, movie_title = json.loads(movie_data)
        await self.redis_client.sadd(f"{SEEN_KEY_PREFIX}{user_id}", movie_id)

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
                timeout=0.1,
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

    def _matches_preferences(self, movie: MovieDetails, prefs: UserPreferences) -> bool:
        if prefs.included_genres:
            if not set(movie.genres) & set(prefs.included_genres):
                return False

        if prefs.excluded_genres:
            if set(movie.genres) & set(prefs.excluded_genres):
                return False

        if prefs.min_release_year and movie.release_year < prefs.min_release_year:
            return False

        if prefs.max_release_year and movie.release_year > prefs.max_release_year:
            return False

        if prefs.min_rating is not None and movie.rating < prefs.min_rating:
            return False

        if not prefs.include_adult and movie.is_adult:
            return False

        if prefs.movie_providers:
            pref_names = {p.name for p in prefs.movie_providers}
            movie_names = {p.name for p in movie.movie_providers}
            if not movie_names & pref_names:
                return False

        return True

    async def refill_queue(
        self,
        user_id: int,
        queue_key: str,
        user_preferences: Optional[UserPreferences] = None,
    ):
        logger.info("Refilling queue for user %s", user_id)

        await self.redis_client.delete(queue_key)

        batch_size = self.settings.app_logic.batch_size
        # With filters active we may need to look deep into the ranked list to find
        # movies that pass. Retry with 2x candidates each round (new movies only).
        fetch_n = batch_size * (
            self.settings.app_logic.over_fetch_factor if user_preferences else 1
        )
        max_candidates = batch_size * 20

        already_seen: set[int] = set()
        items: list[str] = []

        while True:
            ranked_movie_ids = await self.recommender.get_top_n_recommendations(
                user_id=user_id, n=fetch_n
            )

            new_ids = [mid for mid in ranked_movie_ids if mid not in already_seen]
            if not new_ids:
                break  # recommender has no more candidates

            movies = [
                (
                    mid,
                    self.recommender.model_artifacts.movie_id_to_title.get(
                        mid, f"movie_{mid}"
                    ),
                )
                for mid in new_ids
            ]
            logger.info(
                "Got %d new candidates (total fetched so far: %d)",
                len(movies),
                len(already_seen),
            )

            hydrated = await asyncio.gather(
                *(self.hydrator.get_or_fetch_movie(mid, title) for mid, title in movies)
            )

            for (movie_id, movie_title), result in zip(movies, hydrated):
                already_seen.add(movie_id)
                if result is None:
                    continue
                if user_preferences and not self._matches_preferences(
                    result, user_preferences
                ):
                    continue
                items.append(json.dumps([movie_id, movie_title]))

            # Stop once we have a full batch, have no filters, or exhausted the cap
            if items or not user_preferences or len(already_seen) >= max_candidates:
                break

            fetch_n = min(fetch_n * 2, max_candidates)

        logger.info(
            "Queue for user %s: %d movies after filtering (%d candidates tried)",
            user_id,
            len(items),
            len(already_seen),
        )
        if items:
            await self.redis_client.rpush(queue_key, *items)
