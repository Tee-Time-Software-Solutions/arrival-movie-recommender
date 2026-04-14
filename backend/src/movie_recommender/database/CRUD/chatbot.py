"""READ-ONLY CRUD operations for the chatbot feature."""

from collections import defaultdict

from sqlalchemy import func, select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.models import (
    genres,
    movies,
    movies_genres,
    swipes,
)


async def search_movies_by_criteria(
    db: AsyncSession,
    genre_names: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    keyword: str | None = None,
    min_rating: float | None = None,
    limit: int = 10,
) -> list[int]:
    """Search movies by genre, year range, keyword, and rating. Returns movie IDs."""
    query = select(movies.c.id).where(movies.c.tmdb_id.isnot(None))

    if genre_names:
        query = query.where(
            movies.c.id.in_(
                select(movies_genres.c.movie_id)
                .join(genres, genres.c.id == movies_genres.c.genre_id)
                .where(genres.c.name.in_(genre_names))
            )
        )

    if min_year is not None:
        query = query.where(movies.c.release_year >= min_year)

    if max_year is not None:
        query = query.where(movies.c.release_year <= max_year)

    if keyword:
        query = query.where(
            or_(
                movies.c.title.ilike(f"%{keyword}%"),
                movies.c.synopsis.ilike(f"%{keyword}%"),
            )
        )

    if min_rating is not None:
        query = query.where(movies.c.tmdb_rating >= min_rating)

    query = query.order_by(movies.c.tmdb_rating.desc().nulls_last()).limit(limit)

    result = await db.execute(query)
    return [row.id for row in result]


async def get_user_taste_profile(
    db: AsyncSession,
    user_id: int,
) -> dict:
    """Aggregate the user's swipe history into a taste profile summary."""

    # Count likes and dislikes
    counts_result = await db.execute(
        select(
            swipes.c.action_type,
            func.count().label("cnt"),
        )
        .where(swipes.c.user_id == user_id)
        .where(swipes.c.action_type.in_(["like", "dislike"]))
        .group_by(swipes.c.action_type)
    )
    counts = {row.action_type: row.cnt for row in counts_result}
    total_likes = counts.get("like", 0)
    total_dislikes = counts.get("dislike", 0)

    # Genre counts for liked movies
    genre_result = await db.execute(
        select(genres.c.name, func.count().label("cnt"))
        .select_from(
            swipes.join(movies, swipes.c.movie_id == movies.c.id)
            .join(movies_genres, movies.c.id == movies_genres.c.movie_id)
            .join(genres, genres.c.id == movies_genres.c.genre_id)
        )
        .where(swipes.c.user_id == user_id, swipes.c.action_type == "like")
        .group_by(genres.c.name)
        .order_by(func.count().desc())
    )
    genre_counts = [{"genre": row.name, "count": row.cnt} for row in genre_result]

    # Top 10 liked movies with details
    liked_movies_result = await db.execute(
        select(
            movies.c.title,
            movies.c.release_year,
            movies.c.tmdb_rating,
        )
        .join(swipes, swipes.c.movie_id == movies.c.id)
        .where(swipes.c.user_id == user_id, swipes.c.action_type == "like")
        .order_by(movies.c.tmdb_rating.desc().nulls_last())
        .limit(10)
    )
    top_movies = []
    movie_years = []
    for row in liked_movies_result:
        top_movies.append(
            {
                "title": row.title,
                "year": row.release_year,
                "rating": float(row.tmdb_rating) if row.tmdb_rating else None,
            }
        )
        if row.release_year:
            movie_years.append(row.release_year)

    # Genres per top movie
    if top_movies:
        liked_ids_result = await db.execute(
            select(swipes.c.movie_id)
            .where(swipes.c.user_id == user_id, swipes.c.action_type == "like")
        )
        liked_ids = [r.movie_id for r in liked_ids_result]

        genre_map_result = await db.execute(
            select(movies_genres.c.movie_id, genres.c.name)
            .join(genres, genres.c.id == movies_genres.c.genre_id)
            .where(movies_genres.c.movie_id.in_(liked_ids))
        )
        genres_by_movie: dict[int, list[str]] = defaultdict(list)
        for row in genre_map_result:
            genres_by_movie[row.movie_id].append(row.name)

    # Year range and average rating of liked movies
    stats_result = await db.execute(
        select(
            func.min(movies.c.release_year).label("min_year"),
            func.max(movies.c.release_year).label("max_year"),
            func.avg(movies.c.tmdb_rating).label("avg_rating"),
        )
        .join(swipes, swipes.c.movie_id == movies.c.id)
        .where(swipes.c.user_id == user_id, swipes.c.action_type == "like")
    )
    stats = stats_result.first()

    return {
        "total_likes": total_likes,
        "total_dislikes": total_dislikes,
        "genre_counts": genre_counts,
        "top_movies": top_movies,
        "year_range": {
            "min": stats.min_year if stats else None,
            "max": stats.max_year if stats else None,
        },
        "avg_rating": round(float(stats.avg_rating), 2) if stats and stats.avg_rating else None,
    }
