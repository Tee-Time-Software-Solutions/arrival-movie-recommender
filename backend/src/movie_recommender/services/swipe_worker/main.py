import logging

from movie_recommender.database.CRUD.interactions import create_swipe
from movie_recommender.database.engine import DatabaseEngine

logger = logging.getLogger(__name__)


async def persist_swipe(
    user_id: int,
    movie_id: int,
    action_type: str,
    is_supercharged: bool,
) -> None:
    """Fire-and-forget task that writes a single swipe event to the DB."""
    try:
        async with DatabaseEngine().session_factory() as db:
            await create_swipe(
                db=db,
                user_id=user_id,
                movie_id=movie_id,
                action_type=action_type,
                is_supercharged=is_supercharged,
            )
    except Exception:
        logger.exception(
            "Failed to persist swipe event: user=%s movie=%s", user_id, movie_id
        )
