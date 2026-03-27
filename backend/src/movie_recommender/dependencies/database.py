from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.engine import DatabaseEngine


async def get_db() -> AsyncSession:
    async with DatabaseEngine().session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
