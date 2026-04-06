from collections import defaultdict

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.models import (
    MovieRow,
    movies,
    genres,
    movies_genres,
    crew_person,
    movies_cast_crew,
    providers,
    movies_providers,
)
from movie_recommender.schemas.requests.movies import (
    MovieDetails,
    CastMember,
    MovieProvider as MovieProviderSchema,
)


async def get_movie_by_id(db: AsyncSession, movie_id: int) -> MovieRow | None:
    result = await db.execute(select(movies).where(movies.c.id == movie_id))
    return result.first()



async def save_hydrated_movie(
    db: AsyncSession, movie_db_id: int, details: MovieDetails
):
    """Persist full TMDB metadata for a movie. Updates existing stub or creates new."""
    movie_values = dict(
        tmdb_id=details.tmdb_id,
        title=details.title,
        poster_url=details.poster_url,
        release_year=details.release_year,
        tmdb_rating=details.rating,
        synopsis=details.synopsis,
        runtime=details.runtime,
        is_adult=details.is_adult,
        trailer_url=str(details.trailer_url) if details.trailer_url else None,
    )

    existing = await db.execute(select(movies.c.id).where(movies.c.id == movie_db_id))
    if existing.first():
        await db.execute(
            update(movies).where(movies.c.id == movie_db_id).values(**movie_values)
        )
    else:
        await db.execute(insert(movies).values(id=movie_db_id, **movie_values))

    for genre_name in details.genres:
        genre_id = await _get_or_create_genre(db, genre_name)
        exists = await db.execute(
            select(movies_genres).where(
                movies_genres.c.movie_id == movie_db_id,
                movies_genres.c.genre_id == genre_id,
            )
        )
        if not exists.first():
            await db.execute(
                insert(movies_genres).values(movie_id=movie_db_id, genre_id=genre_id)
            )

    for member in details.cast:
        crew_id = await _get_or_create_crew_person(db, member)
        exists = await db.execute(
            select(movies_cast_crew).where(
                movies_cast_crew.c.movie_id == movie_db_id,
                movies_cast_crew.c.crew_person_id == crew_id,
            )
        )
        if not exists.first():
            await db.execute(
                insert(movies_cast_crew).values(
                    movie_id=movie_db_id, crew_person_id=crew_id
                )
            )

    for prov in details.movie_providers:
        provider_id = await _get_or_create_provider(
            db, prov.name, prov.provider_type.value
        )
        exists = await db.execute(
            select(movies_providers).where(
                movies_providers.c.movie_id == movie_db_id,
                movies_providers.c.provider_id == provider_id,
            )
        )
        if not exists.first():
            await db.execute(
                insert(movies_providers).values(
                    movie_id=movie_db_id, provider_id=provider_id
                )
            )

    await db.commit()


async def movie_to_details(db: AsyncSession, movie_id: int) -> MovieDetails:
    """Fetch movie + all relations and build a MovieDetails response."""
    result = await db.execute(select(movies).where(movies.c.id == movie_id))
    m = result.first()

    genre_rows = await db.execute(
        select(genres.c.name)
        .join(movies_genres, genres.c.id == movies_genres.c.genre_id)
        .where(movies_genres.c.movie_id == movie_id)
    )

    cast_rows = await db.execute(
        select(crew_person)
        .join(movies_cast_crew, crew_person.c.id == movies_cast_crew.c.crew_person_id)
        .where(movies_cast_crew.c.movie_id == movie_id)
    )

    provider_rows = await db.execute(
        select(providers)
        .join(movies_providers, providers.c.id == movies_providers.c.provider_id)
        .where(movies_providers.c.movie_id == movie_id)
    )

    return MovieDetails(
        movie_db_id=m.id,
        tmdb_id=m.tmdb_id,
        title=m.title,
        poster_url=m.poster_url or "",
        release_year=m.release_year or 0,
        rating=m.tmdb_rating or 0.0,
        genres=[row.name for row in genre_rows],
        is_adult=m.is_adult or False,
        synopsis=m.synopsis or "",
        runtime=m.runtime or 0,
        trailer_url=m.trailer_url,
        cast=[
            CastMember(
                name=row.name,
                role_type=row.role_type,
                character_name=row.character_name,
                profile_path=row.image_url,
            )
            for row in cast_rows
        ],
        movie_providers=[
            MovieProviderSchema(name=row.name, provider_type=row.provider_type)
            for row in provider_rows
        ],
    )


async def movies_to_details_bulk(
    db: AsyncSession, movie_ids: list[int]
) -> list[MovieDetails]:
    """Fetch multiple movies with all relations in bulk (avoids N+1 queries)."""
    if not movie_ids:
        return []

    movie_rows = await db.execute(select(movies).where(movies.c.id.in_(movie_ids)))
    movies_by_id = {row.id: row for row in movie_rows}

    genre_rows = await db.execute(
        select(movies_genres.c.movie_id, genres.c.name)
        .join(genres, genres.c.id == movies_genres.c.genre_id)
        .where(movies_genres.c.movie_id.in_(movie_ids))
    )
    genres_by_movie: dict[int, list[str]] = defaultdict(list)
    for row in genre_rows:
        genres_by_movie[row.movie_id].append(row.name)

    cast_rows = await db.execute(
        select(movies_cast_crew.c.movie_id, crew_person)
        .join(crew_person, crew_person.c.id == movies_cast_crew.c.crew_person_id)
        .where(movies_cast_crew.c.movie_id.in_(movie_ids))
    )
    cast_by_movie: dict[int, list[CastMember]] = defaultdict(list)
    for row in cast_rows:
        cast_by_movie[row.movie_id].append(
            CastMember(
                name=row.name,
                role_type=row.role_type,
                character_name=row.character_name,
                profile_path=row.image_url,
            )
        )

    provider_rows = await db.execute(
        select(movies_providers.c.movie_id, providers)
        .join(providers, providers.c.id == movies_providers.c.provider_id)
        .where(movies_providers.c.movie_id.in_(movie_ids))
    )
    providers_by_movie: dict[int, list[MovieProviderSchema]] = defaultdict(list)
    for row in provider_rows:
        providers_by_movie[row.movie_id].append(
            MovieProviderSchema(name=row.name, provider_type=row.provider_type)
        )

    results = []
    for mid in movie_ids:
        m = movies_by_id.get(mid)
        if not m:
            continue
        results.append(
            MovieDetails(
                movie_db_id=m.id,
                tmdb_id=m.tmdb_id,
                title=m.title,
                poster_url=m.poster_url or "",
                release_year=m.release_year or 0,
                rating=m.tmdb_rating or 0.0,
                genres=genres_by_movie.get(mid, []),
                is_adult=m.is_adult or False,
                synopsis=m.synopsis or "",
                runtime=m.runtime or 0,
                trailer_url=m.trailer_url,
                cast=cast_by_movie.get(mid, []),
                movie_providers=providers_by_movie.get(mid, []),
            )
        )
    return results


async def _get_or_create_crew_person(db: AsyncSession, member: CastMember) -> int:
    """Find crew person by tmdb_person_id (preferred) or create new."""
    if member.tmdb_person_id:
        result = await db.execute(
            select(crew_person.c.id).where(
                crew_person.c.tmdb_person_id == member.tmdb_person_id
            )
        )
        row = result.first()
        if row:
            return row.id

    result = await db.execute(
        insert(crew_person)
        .values(
            tmdb_person_id=member.tmdb_person_id,
            name=member.name,
            role_type=member.role_type,
            character_name=member.character_name,
            image_url=str(member.profile_path) if member.profile_path else None,
        )
        .returning(crew_person.c.id)
    )
    return result.scalar_one()


async def _get_or_create_genre(db: AsyncSession, name: str) -> int:
    result = await db.execute(select(genres.c.id).where(genres.c.name == name))
    row = result.first()
    if row:
        return row.id
    result = await db.execute(insert(genres).values(name=name).returning(genres.c.id))
    return result.scalar_one()


async def _get_or_create_provider(
    db: AsyncSession, name: str, provider_type: str
) -> int:
    result = await db.execute(
        select(providers.c.id).where(
            providers.c.name == name, providers.c.provider_type == provider_type
        )
    )
    row = result.first()
    if row:
        return row.id
    result = await db.execute(
        insert(providers)
        .values(name=name, provider_type=provider_type)
        .returning(providers.c.id)
    )
    return result.scalar_one()
