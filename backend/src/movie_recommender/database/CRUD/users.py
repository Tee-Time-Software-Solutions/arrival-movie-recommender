from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.models import User, Preference
from movie_recommender.schemas.requests.users import UserCreate


async def create_user(db: AsyncSession, user_data: UserCreate) -> User:
    user = User(
        firebase_uid=user_data.firebase_uid,
        profile_image_url=user_data.profile_image_url,
        email=user_data.email,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def get_user_by_firebase_uid(db: AsyncSession, firebase_uid: str) -> User | None:
    result = await db.execute(select(User).where(User.firebase_uid == firebase_uid))
    return result.scalar_one_or_none()


async def get_user_preferences(db: AsyncSession, user_id: int) -> Preference | None:
    result = await db.execute(select(Preference).where(Preference.user_id == user_id))
    return result.scalar_one_or_none()


async def update_user_preferences(
    db: AsyncSession,
    user_id: int,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = None,
) -> Preference:
    result = await db.execute(select(Preference).where(Preference.user_id == user_id))
    pref = result.scalar_one_or_none()
    if not pref:
        pref = Preference(user_id=user_id)
        db.add(pref)

    if min_year is not None:
        pref.min_year = min_year
    if max_year is not None:
        pref.max_year = max_year
    if min_rating is not None:
        pref.min_rating = min_rating

    await db.commit()
    await db.refresh(pref)
    return pref
