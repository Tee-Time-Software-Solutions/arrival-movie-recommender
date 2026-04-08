import asyncio

import redis
from fastapi import APIRouter, Depends, HTTPException
from neo4j import AsyncDriver
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.CRUD.movies import get_movie_by_id
from movie_recommender.database.CRUD.users import get_user_by_firebase_uid
from movie_recommender.dependencies.database import get_db
from movie_recommender.dependencies.firebase import verify_user
from movie_recommender.dependencies.neo4j import get_neo4j_driver
from movie_recommender.dependencies.recommender import get_recommender
from movie_recommender.dependencies.redis import get_async_redis
from movie_recommender.schemas.requests.interactions import (
    RegisteredFeedback,
    SwipeAction,
    SwipeRequest,
)
from movie_recommender.services.knowledge_graph.beacon import update_beacon_on_swipe
from movie_recommender.services.recommender.main import Recommender
from movie_recommender.services.swipe_worker.main import enqueue_swipe

router = APIRouter(prefix="/interactions")


@router.post(path="/{movie_id}/swipe")
async def register_movie_interaction(
    movie_id: int,
    swipe_data: SwipeRequest,
    db: AsyncSession = Depends(get_db),
    recommender: Recommender = Depends(get_recommender),
    redis_client: redis.Redis = Depends(get_async_redis),
    neo4j_driver: AsyncDriver = Depends(get_neo4j_driver),
    auth_user=Depends(verify_user()),
) -> RegisteredFeedback:
    """
    1. Validate movie exists in DB
    2. Enqueue swipe event to Redis (background worker persists to DB)
    3. Trigger recommender update
    """
    if swipe_data.action_type == SwipeAction.SKIP and swipe_data.is_supercharged:
        raise HTTPException(
            status_code=400, detail="Cannot have 'SKIP' interaction supercharged"
        )

    user_row = await get_user_by_firebase_uid(db, auth_user["uid"])
    if user_row is None:
        raise HTTPException(status_code=404, detail="User not found")

    movie_row = await get_movie_by_id(db, movie_id)
    if movie_row is None:
        raise HTTPException(status_code=404, detail="Movie not found")

    await enqueue_swipe(
        redis_client=redis_client,
        user_id=user_row.id,
        movie_id=movie_id,
        action_type=swipe_data.action_type.value,
        is_supercharged=swipe_data.is_supercharged,
    )

    await recommender.set_user_feedback(
        user_id=user_row.id,
        movie_id=movie_id,
        interaction_type=swipe_data.action_type,
        is_supercharged=swipe_data.is_supercharged,
    )

    # Fire-and-forget beacon map update (KG explainability)
    if movie_row.tmdb_id:
        asyncio.create_task(
            update_beacon_on_swipe(
                neo4j_driver=neo4j_driver,
                redis_client=redis_client,
                user_id=user_row.id,
                movie_tmdb_id=movie_row.tmdb_id,
                action_type=swipe_data.action_type.value,
                is_supercharged=swipe_data.is_supercharged,
            )
        )

    return RegisteredFeedback(
        interaction_id=0,
        movie_id=movie_id,
        action_type=swipe_data.action_type,
        is_supercharged=swipe_data.is_supercharged,
        registered=True,
    )
