import asyncio
import json
import logging

import redis.asyncio as aioredis

from movie_recommender.database.CRUD.interactions import create_swipe

logger = logging.getLogger(__name__)

SWIPE_QUEUE_KEY = "swipe_events"


async def enqueue_swipe(
    redis_client: aioredis.Redis,
    user_id: int,
    movie_id: int,
    action_type: str,
    is_supercharged: bool,
) -> None:
    payload = json.dumps(
        {
            "user_id": user_id,
            "movie_id": movie_id,
            "action_type": action_type,
            "is_supercharged": is_supercharged,
        }
    )
    await redis_client.rpush(SWIPE_QUEUE_KEY, payload)


async def drain_swipe_queue(
    redis_client: aioredis.Redis,
    db_session_factory,
    batch_size: int = 50,
    poll_interval: float = 0.5,
) -> None:
    """Background loop that drains swipe events from Redis into the DB."""
    logger.info("Swipe worker started")
    while True:
        events = []
        for _ in range(batch_size):
            raw = await redis_client.lpop(SWIPE_QUEUE_KEY)
            if raw is None:
                break
            events.append(json.loads(raw))

        if events:
            async with db_session_factory() as db:
                for event in events:
                    try:
                        await create_swipe(
                            db=db,
                            user_id=event["user_id"],
                            movie_id=event["movie_id"],
                            action_type=event["action_type"],
                            is_supercharged=event["is_supercharged"],
                        )
                    except Exception:
                        logger.exception("Failed to persist swipe event: %s", event)
            logger.debug("Persisted %d swipe events", len(events))

        await asyncio.sleep(poll_interval)
