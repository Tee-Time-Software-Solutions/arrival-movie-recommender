from sqlalchemy import and_, case, insert, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.models import SwipeRow, swipes

SUPERCHARGED_SCORE = 2
LIKE_SCORE = 1
DISLIKE_SCORE = -1


async def get_all_swipes(db: AsyncSession) -> list[SwipeRow]:
    """Return every swipe row for offline pipeline export."""
    result = await db.execute(
        select(
            swipes.c.user_id,
            swipes.c.movie_id,
            swipes.c.action_type,
            swipes.c.is_supercharged,
            swipes.c.created_at,
        )
    )
    return [row._asdict() for row in result]


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


async def create_swipes_bulk(
    db: AsyncSession,
    user_id: int,
    movie_ids: list[int],
    action_type: str,
    is_supercharged: bool = False,
) -> None:
    """Insert multiple swipe records in a single statement. Caller must commit."""
    if not movie_ids:
        return
    await db.execute(
        insert(swipes).values(
            [
                {
                    "user_id": user_id,
                    "movie_id": mid,
                    "action_type": action_type,
                    "is_supercharged": is_supercharged,
                }
                for mid in movie_ids
            ]
        )
    )


async def get_user_liked_movies(
    db: AsyncSession,
    user_id: int,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[int], int]:
    """Return (movie_ids, total_count) for movies with positive engagement score."""
    weight = case(
        (
            and_(
                swipes.c.action_type == "like",
                swipes.c.is_supercharged.is_(True),
            ),
            SUPERCHARGED_SCORE,
        ),
        (swipes.c.action_type == "like", LIKE_SCORE),
        (
            and_(
                swipes.c.action_type == "dislike",
                swipes.c.is_supercharged.is_(True),
            ),
            SUPERCHARGED_SCORE * -1,
        ),
        (swipes.c.action_type == "dislike", DISLIKE_SCORE),
        else_=0,
    )

    score = func.sum(weight).label("score")

    base = (
        select(swipes.c.movie_id, score)
        .where(swipes.c.user_id == user_id)
        .group_by(swipes.c.movie_id)
        .having(func.sum(weight) > 0)
    )

    count_result = await db.execute(select(func.count()).select_from(base.subquery()))
    total = count_result.scalar_one()

    result = await db.execute(
        base.order_by(score.desc(), func.max(swipes.c.created_at).desc())
        .limit(limit)
        .offset(offset)
    )
    movie_ids = [row.movie_id for row in result]
    return movie_ids, total


async def get_all_rated_movies(
    db: AsyncSession,
    user_id: int,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[int], int]:
    """Return (movie_ids, total_count) for all rated movies (likes + dislikes), ranked by score."""
    weight = case(
        (
            and_(
                swipes.c.action_type == "like",
                swipes.c.is_supercharged.is_(True),
            ),
            SUPERCHARGED_SCORE,
        ),
        (swipes.c.action_type == "like", LIKE_SCORE),
        (
            and_(
                swipes.c.action_type == "dislike",
                swipes.c.is_supercharged.is_(True),
            ),
            SUPERCHARGED_SCORE * -1,
        ),
        (swipes.c.action_type == "dislike", DISLIKE_SCORE),
        else_=0,
    )

    score = func.sum(weight).label("score")

    base = (
        select(swipes.c.movie_id, score)
        .where(
            swipes.c.user_id == user_id,
            swipes.c.action_type.in_(["like", "dislike"]),
        )
        .group_by(swipes.c.movie_id)
    )

    count_result = await db.execute(select(func.count()).select_from(base.subquery()))
    total = count_result.scalar_one()

    result = await db.execute(
        base.order_by(score.desc(), func.max(swipes.c.created_at).desc())
        .limit(limit)
        .offset(offset)
    )
    movie_ids = [row.movie_id for row in result]
    return movie_ids, total
