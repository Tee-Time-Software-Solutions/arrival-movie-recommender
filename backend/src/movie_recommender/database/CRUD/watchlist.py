from sqlalchemy import delete, func, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.models import watchlist


async def add_to_watchlist(
    db: AsyncSession,
    user_id: int,
    movie_id: int,
):
    try:
        result = await db.execute(
            insert(watchlist)
            .values(user_id=user_id, movie_id=movie_id)
            .returning(*watchlist.c)
        )
        await db.commit()
        return result.first()
    except IntegrityError:
        await db.rollback()
        return None


async def remove_from_watchlist(
    db: AsyncSession,
    user_id: int,
    movie_id: int,
) -> bool:
    result = await db.execute(
        delete(watchlist).where(
            watchlist.c.user_id == user_id,
            watchlist.c.movie_id == movie_id,
        )
    )
    await db.commit()
    return result.rowcount > 0


async def get_user_watchlist(
    db: AsyncSession,
    user_id: int,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[int], int]:
    """Return (movie_ids, total_count) for the user's watchlist, newest first."""
    base = select(watchlist.c.movie_id).where(watchlist.c.user_id == user_id)

    count_result = await db.execute(select(func.count()).select_from(base.subquery()))
    total = count_result.scalar_one()

    result = await db.execute(
        base.order_by(watchlist.c.added_at.desc()).limit(limit).offset(offset)
    )
    movie_ids = [row.movie_id for row in result]
    return movie_ids, total
