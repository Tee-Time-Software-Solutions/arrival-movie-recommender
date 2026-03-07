from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.models import Swipe


async def create_swipe(
    db: AsyncSession,
    user_id: int,
    movie_id: int,
    action_type: str,
    is_supercharged: bool = False,
) -> Swipe:
    swipe = Swipe(
        user_id=user_id,
        movie_id=movie_id,
        action_type=action_type,
        is_supercharged=is_supercharged,
    )
    db.add(swipe)
    await db.commit()
    await db.refresh(swipe)
    return swipe
