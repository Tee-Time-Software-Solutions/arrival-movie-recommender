import numpy as np
from sqlalchemy import func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.models import user_online_vectors


async def get_user_vector(db: AsyncSession, user_id: int) -> np.ndarray | None:
    result = await db.execute(
        select(user_online_vectors.c.vector).where(
            user_online_vectors.c.user_id == user_id
        )
    )
    row = result.first()
    if row is None:
        return None
    return np.array(row.vector, dtype=np.float32)


async def save_user_vector(db: AsyncSession, user_id: int, vector: np.ndarray) -> None:
    existing = await db.execute(
        select(user_online_vectors.c.user_id).where(
            user_online_vectors.c.user_id == user_id
        )
    )
    if existing.first():
        await db.execute(
            update(user_online_vectors)
            .where(user_online_vectors.c.user_id == user_id)
            .values(vector=vector.tolist(), updated_at=func.now())
        )
    else:
        await db.execute(
            insert(user_online_vectors).values(user_id=user_id, vector=vector.tolist())
        )
    await db.commit()
