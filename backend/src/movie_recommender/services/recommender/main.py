import asyncio
import logging
from typing import Callable, List, Optional  # List kept for return type annotations

import numpy as np
import redis.asyncio as aioredis

from movie_recommender.core.settings.main import AppSettings
from movie_recommender.database.CRUD.user_vectors import (
    get_user_vector,
    save_user_vector,
)
from movie_recommender.schemas.requests.interactions import SwipeAction
from movie_recommender.services.recommender.pipeline.online.artifacts import (
    RecommenderArtifacts,
    load_model_artifacts,
)

from movie_recommender.services.recommender.pipeline.online.learning.feedback import (
    apply_feedback_update,
)
from movie_recommender.services.recommender.pipeline.online.serving.ranker import (
    rank_movie_ids,
)
from movie_recommender.services.recommender.pipeline.online.serving.user_state import (
    cold_start_vector,
)

logger = logging.getLogger(__name__)

USER_VECTOR_KEY_PREFIX = "user_vector:"
SEEN_KEY_PREFIX = "seen:user:"


class Recommender:
    def __init__(self, db_session_factory: Callable) -> None:
        settings = AppSettings()
        self.learning_rate = settings.app_logic.learning_rate
        self.norm_cap = settings.app_logic.norm_cap
        self.model_artifacts: RecommenderArtifacts = load_model_artifacts()
        self._db_session_factory = db_session_factory
        self._redis: Optional[aioredis.Redis] = None

    def set_redis(self, redis_client: aioredis.Redis) -> None:
        self._redis = redis_client

    async def _get_user_vector(self, user_id: int) -> np.ndarray:
        # 1. Redis hot cache
        if self._redis:
            raw = await self._redis.get(f"{USER_VECTOR_KEY_PREFIX}{user_id}")
            if raw:
                return np.frombuffer(raw, dtype=np.float32).copy()

        # 2. Postgres persistent store
        async with self._db_session_factory() as db:
            vector = await get_user_vector(db, user_id)
        if vector is not None:
            if self._redis:
                await self._redis.set(
                    f"{USER_VECTOR_KEY_PREFIX}{user_id}", vector.tobytes()
                )
            return vector

        # 3. Cold start
        return cold_start_vector(self.model_artifacts)

    async def _persist_vector_to_db(self, user_id: int, vector: np.ndarray) -> None:
        try:
            async with self._db_session_factory() as db:
                await save_user_vector(db, user_id, vector)
        except Exception:
            logger.warning(
                "Failed to persist user vector for user %s", user_id, exc_info=True
            )

    async def get_top_n_recommendations(self, user_id: int, n: int) -> List[int]:
        user_vector = await self._get_user_vector(user_id)

        seen_movie_ids: set[int] = set()
        if self._redis:
            members = await self._redis.smembers(f"{SEEN_KEY_PREFIX}{user_id}")
            seen_movie_ids = {int(m) for m in members} if members else set()

        return rank_movie_ids(
            n=n,
            model_artifacts=self.model_artifacts,
            user_vector=user_vector,
            seen_movie_ids=seen_movie_ids,
        )

    async def set_user_feedback(
        self,
        user_id: int,
        movie_id: int,
        interaction_type: SwipeAction,
        is_supercharged: bool,
    ) -> None:
        user_vector = await self._get_user_vector(user_id)

        updated = apply_feedback_update(
            model_artifacts=self.model_artifacts,
            user_vector=user_vector,
            movie_id=movie_id,
            interaction_type=interaction_type,
            is_supercharged=is_supercharged,
            learning_rate=self.learning_rate,
            norm_cap=self.norm_cap,
        )

        if updated is not None:
            if self._redis:
                await self._redis.set(
                    f"{USER_VECTOR_KEY_PREFIX}{user_id}", updated.tobytes()
                )
            asyncio.create_task(self._persist_vector_to_db(user_id, updated))
