import asyncio
import logging
from typing import Callable, List, Optional  # List kept for return type annotations

import numpy as np
import redis.asyncio as aioredis
from neo4j import AsyncDriver

from movie_recommender.core.settings.main import AppSettings
from movie_recommender.database.CRUD.user_vectors import (
    get_user_vector,
    save_user_vector,
)
from movie_recommender.schemas.requests.interactions import SwipeAction
from movie_recommender.services.knowledge_graph.beacon import load_beacon_map
from movie_recommender.services.recommender.pipeline.online.artifacts import (
    RecommenderArtifacts,
    load_model_artifacts,
)

from movie_recommender.services.recommender.pipeline.online.learning.feedback import (
    apply_feedback_update,
)
from movie_recommender.services.recommender.pipeline.online.learning.adaptive import (
    adaptive_learning_rate,
    counts_for_adaptation,
    get_feedback_count,
    increment_feedback_count,
)
from movie_recommender.services.recommender.pipeline.online.exploration import (
    get_genre_impression_counts,
)
from movie_recommender.services.recommender.pipeline.online.serving.graph_rerank import (
    als_shortlist,
    blend_scores,
    compute_graph_scores,
)
from movie_recommender.services.recommender.pipeline.online.serving.ranker import (
    score_candidates,
    select_top_n,
)
from movie_recommender.services.recommender.pipeline.online.serving.user_state import (
    base_user_vector,
    cold_start_vector,
)

logger = logging.getLogger(__name__)

USER_VECTOR_KEY_PREFIX = "user_vector:"
SEEN_KEY_PREFIX = "seen:user:"


class Recommender:
    def __init__(self, db_session_factory: Callable) -> None:
        settings = AppSettings()
        self.learning_rate = settings.app_logic.learning_rate
        self.adaptive_learning_strength = settings.app_logic.adaptive_learning_strength
        self.norm_cap = settings.app_logic.norm_cap
        self.exploration_weight = settings.app_logic.exploration_weight
        self.diversity_weight = settings.app_logic.diversity_weight
        self.graph_weight = settings.app_logic.graph_weight
        self.graph_rerank_top_k = settings.app_logic.graph_rerank_top_k
        self.model_artifacts: RecommenderArtifacts = load_model_artifacts()
        self._db_session_factory = db_session_factory
        self._redis: Optional[aioredis.Redis] = None
        self._neo4j_driver: Optional[AsyncDriver] = None

    def set_redis(self, redis_client: aioredis.Redis) -> None:
        self._redis = redis_client

    def set_neo4j_driver(self, driver: AsyncDriver) -> None:
        self._neo4j_driver = driver

    async def _get_user_vector(self, user_id: int) -> np.ndarray:
        # 1. Redis hot cache
        if self._redis:
            raw = await self._redis.get(f"{USER_VECTOR_KEY_PREFIX}{user_id}")
            if raw:
                return np.frombuffer(raw, dtype=np.float32).copy()

        # 2. Postgres persistent store (updated vectors from online learning)
        async with self._db_session_factory() as db:
            vector = await get_user_vector(db, user_id)
        if vector is not None:
            if self._redis:
                await self._redis.set(
                    f"{USER_VECTOR_KEY_PREFIX}{user_id}", vector.tobytes()
                )
            return vector

        # 3. ALS-trained embedding for users who were in the training set
        # 4. Cold start (mean of all users) for genuinely new users
        return base_user_vector(self.model_artifacts, user_id)

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
        genre_impression_counts: dict[str, int] = {}
        if self._redis:
            members = await self._redis.smembers(f"{SEEN_KEY_PREFIX}{user_id}")
            seen_movie_ids = {int(m) for m in members} if members else set()
            genre_impression_counts = await get_genre_impression_counts(
                self._redis, user_id
            )

        candidate_ids, candidate_embeddings, scores = score_candidates(
            model_artifacts=self.model_artifacts,
            user_vector=user_vector,
            seen_movie_ids=seen_movie_ids,
            genre_impression_counts=genre_impression_counts,
            exploration_weight=self.exploration_weight,
        )

        if (
            self.graph_weight > 0
            and self._neo4j_driver is not None
            and self._redis is not None
            and len(candidate_ids) > 0
        ):
            try:
                await self._apply_graph_rerank(user_id, candidate_ids, scores)
            except Exception:
                logger.warning(
                    "Graph rerank failed for user %s — falling back to ALS",
                    user_id,
                    exc_info=True,
                )

        return select_top_n(
            candidate_ids=candidate_ids,
            candidate_embeddings=candidate_embeddings,
            scores=scores,
            n=n,
            diversity_weight=self.diversity_weight,
        )

    async def _apply_graph_rerank(
        self,
        user_id: int,
        candidate_ids: np.ndarray,
        scores: np.ndarray,
    ) -> None:
        """Mutate `scores` in place: blend ALS with KG-affinity for the top-K shortlist."""
        beacon_map = await load_beacon_map(self._redis, user_id)
        if not beacon_map:
            return

        movie_id_to_tmdb_id = self.model_artifacts.movie_id_to_tmdb_id
        if not movie_id_to_tmdb_id:
            return

        shortlist_idx, shortlist_ids = als_shortlist(
            scores, candidate_ids, self.graph_rerank_top_k
        )

        # Map internal movie_ids to tmdb_ids; drop entries without a tmdb_id.
        shortlist_tmdb_ids: list[int] = []
        idx_to_tmdb: list[tuple[int, int]] = []  # (position-in-shortlist, tmdb_id)
        for pos, mid in enumerate(shortlist_ids.tolist()):
            tid = movie_id_to_tmdb_id.get(int(mid))
            if tid is not None:
                shortlist_tmdb_ids.append(tid)
                idx_to_tmdb.append((pos, tid))

        if not shortlist_tmdb_ids:
            return

        graph_scores_dict = await compute_graph_scores(
            self._neo4j_driver, shortlist_tmdb_ids, beacon_map
        )
        if not graph_scores_dict:
            return

        graph_array = np.zeros(len(shortlist_idx), dtype=np.float32)
        for pos, tid in idx_to_tmdb:
            graph_array[pos] = graph_scores_dict.get(tid, 0.0)

        blended = blend_scores(scores[shortlist_idx], graph_array, self.graph_weight)
        scores[shortlist_idx] = blended

    async def set_user_feedback(
        self,
        user_id: int,
        movie_id: int,
        interaction_type: SwipeAction,
        is_supercharged: bool,
    ) -> None:
        user_vector = await self._get_user_vector(user_id)
        effective_learning_rate = self.learning_rate
        if self._redis and counts_for_adaptation(interaction_type):
            feedback_count = await get_feedback_count(self._redis, user_id)
            effective_learning_rate = adaptive_learning_rate(
                base_learning_rate=self.learning_rate,
                feedback_count=feedback_count,
                strength=self.adaptive_learning_strength,
            )

        updated = apply_feedback_update(
            model_artifacts=self.model_artifacts,
            user_vector=user_vector,
            movie_id=movie_id,
            interaction_type=interaction_type,
            is_supercharged=is_supercharged,
            learning_rate=effective_learning_rate,
            norm_cap=self.norm_cap,
        )

        if updated is not None:
            if self._redis:
                await self._redis.set(
                    f"{USER_VECTOR_KEY_PREFIX}{user_id}", updated.tobytes()
                )
                if counts_for_adaptation(interaction_type):
                    await increment_feedback_count(self._redis, user_id)
            asyncio.create_task(self._persist_vector_to_db(user_id, updated))
