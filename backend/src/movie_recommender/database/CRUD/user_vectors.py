import numpy as np
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
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
    vector_list = vector.tolist()
    stmt = (
        pg_insert(user_online_vectors)
        .values(user_id=user_id, vector=vector_list)
        .on_conflict_do_update(
            index_elements=["user_id"],
            set_={"vector": vector_list, "updated_at": func.now()},
        )
    )
    await db.execute(stmt)
    await db.commit()
