from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.models import (
    UserRow,
    PreferenceRow,
    users,
    preferences,
    swipes,
    genres,
    movies_genres,
    excluded_genres,
    included_genres,
)
from movie_recommender.schemas.requests.users import UserCreate


async def create_user(db: AsyncSession, user_data: UserCreate) -> UserRow:
    result = await db.execute(
        insert(users)
        .values(
            firebase_uid=user_data.firebase_uid,
            profile_image_url=user_data.profile_image_url,
            email=user_data.email,
        )
        .returning(*users.c)
    )
    await db.commit()
    return result.first()


async def get_user_by_firebase_uid(
    db: AsyncSession, firebase_uid: str
) -> UserRow | None:
    result = await db.execute(select(users).where(users.c.firebase_uid == firebase_uid))
    return result.first()


async def get_user_preferences(db: AsyncSession, user_id: int) -> PreferenceRow | None:
    result = await db.execute(
        select(preferences).where(preferences.c.user_id == user_id)
    )
    return result.first()


async def get_user_included_genres(db: AsyncSession, user_id: int) -> list[str]:
    result = await db.execute(
        select(genres.c.name)
        .join(included_genres, genres.c.id == included_genres.c.genre_id)
        .where(included_genres.c.user_id == user_id)
    )
    return [row.name for row in result]


async def get_user_excluded_genres(db: AsyncSession, user_id: int) -> list[str]:
    result = await db.execute(
        select(genres.c.name)
        .join(excluded_genres, genres.c.id == excluded_genres.c.genre_id)
        .where(excluded_genres.c.user_id == user_id)
    )
    return [row.name for row in result]


async def get_user_analytics(db: AsyncSession, user_id: int) -> dict:
    """Compute swipe stats and top liked genres for a user."""
    totals = await db.execute(
        select(
            func.count().label("total_swipes"),
            func.count().filter(swipes.c.action_type == "like").label("total_likes"),
            func.count()
            .filter(swipes.c.action_type == "dislike")
            .label("total_dislikes"),
        ).where(swipes.c.user_id == user_id)
    )
    row = totals.first()

    top = await db.execute(
        select(genres.c.name, func.count().label("cnt"))
        .select_from(
            swipes.join(
                movies_genres, swipes.c.movie_id == movies_genres.c.movie_id
            ).join(genres, movies_genres.c.genre_id == genres.c.id)
        )
        .where(swipes.c.user_id == user_id, swipes.c.action_type == "like")
        .group_by(genres.c.name)
        .order_by(func.count().desc())
        .limit(5)
    )

    return {
        "total_swipes": row.total_swipes,
        "total_likes": row.total_likes,
        "total_dislikes": row.total_dislikes,
        "total_seen": row.total_likes + row.total_dislikes,
        "top_genres": [r.name for r in top],
    }


async def update_user_preferences(
    db: AsyncSession,
    user_id: int,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = None,
    include_adult: bool | None = None,
    included_genre_names: list[str] | None = None,
    excluded_genre_names: list[str] | None = None,
):
    existing = await db.execute(
        select(preferences.c.id).where(preferences.c.user_id == user_id)
    )

    values: dict = {}
    if min_year is not None:
        values["min_year"] = min_year
    if max_year is not None:
        values["max_year"] = max_year
    if min_rating is not None:
        values["min_rating"] = min_rating
    if include_adult is not None:
        values["include_adult"] = include_adult

    if existing.first():
        if values:
            await db.execute(
                update(preferences)
                .where(preferences.c.user_id == user_id)
                .values(**values)
            )
    else:
        await db.execute(insert(preferences).values(user_id=user_id, **values))

    if included_genre_names is not None:
        await _sync_genre_list(db, user_id, included_genre_names, included_genres)

    if excluded_genre_names is not None:
        await _sync_genre_list(db, user_id, excluded_genre_names, excluded_genres)

    await db.commit()


async def mark_onboarding_completed(db: AsyncSession, user_id: int) -> None:
    await db.execute(
        update(users).where(users.c.id == user_id).values(onboarding_completed=True)
    )
    await db.commit()


async def _sync_genre_list(
    db: AsyncSession, user_id: int, genre_names: list[str], assoc_table
):
    """Replace all genre associations for a user in the given table."""
    await db.execute(delete(assoc_table).where(assoc_table.c.user_id == user_id))
    for name in genre_names:
        genre_row = await db.execute(select(genres.c.id).where(genres.c.name == name))
        row = genre_row.first()
        if not row:
            result = await db.execute(
                insert(genres).values(name=name).returning(genres.c.id)
            )
            genre_id = result.scalar_one()
        else:
            genre_id = row.id
        await db.execute(insert(assoc_table).values(user_id=user_id, genre_id=genre_id))
