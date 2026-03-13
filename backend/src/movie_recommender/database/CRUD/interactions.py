from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.models import swipes


async def create_swipe(
    db: AsyncSession,
    user_id: int,
    movie_id: int,
    action_type: str,
    is_supercharged: bool = False,
):
    result = await db.execute(
        insert(swipes)
        .values(
            user_id=user_id,
            movie_id=movie_id,
            action_type=action_type,
            is_supercharged=is_supercharged,
        )
        .returning(*swipes.c)
    )
    await db.commit()
    return result.first()
