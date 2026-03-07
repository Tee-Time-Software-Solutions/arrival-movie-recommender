from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from movie_recommender.database.models import (
    Movie,
    Genre,
    MovieGenre,
    CrewPerson,
    MovieCastCrew,
    Provider,
    MovieProvider,
)
from movie_recommender.schemas.requests.movies import (
    MovieDetails,
    CastMember,
    MovieProvider as MovieProviderSchema,
)


async def get_movie_by_id(db: AsyncSession, movie_id: int) -> Movie | None:
    result = await db.execute(
        select(Movie)
        .where(Movie.id == movie_id)
        .options(
            selectinload(Movie.genres),
            selectinload(Movie.cast_crew),
            selectinload(Movie.providers),
        )
    )
    return result.scalar_one_or_none()


async def create_movie_stub(db: AsyncSession, title: str) -> Movie:
    """Create a movie with just a title. Returns the movie with its auto-generated ID."""
    movie = Movie(title=title)
    db.add(movie)
    await db.commit()
    await db.refresh(movie)
    return movie


async def save_hydrated_movie(
    db: AsyncSession, movie_db_id: int, details: MovieDetails
) -> Movie:
    """Persist full TMDB metadata for a movie. Updates existing stub or creates new."""
    movie = await db.get(Movie, movie_db_id)
    if not movie:
        movie = Movie(id=movie_db_id, title=details.title)
        db.add(movie)
        await db.flush()

    movie.tmdb_id = details.tmdb_id
    movie.title = details.title
    movie.poster_url = details.poster_url
    movie.release_year = details.release_year
    movie.tmdb_rating = details.rating
    movie.synopsis = details.synopsis
    movie.runtime = details.runtime
    movie.is_adult = details.is_adult
    movie.trailer_url = str(details.trailer_url) if details.trailer_url else None

    for genre_name in details.genres:
        genre = await _get_or_create_genre(db, genre_name)
        exists = await db.execute(
            select(MovieGenre).where(
                MovieGenre.movie_id == movie.id, MovieGenre.genre_id == genre.id
            )
        )
        if not exists.scalar_one_or_none():
            db.add(MovieGenre(movie_id=movie.id, genre_id=genre.id))

    for member in details.cast:
        crew = CrewPerson(
            name=member.name,
            role_type=member.role_type,
            character_name=member.role_type,
            image_url=str(member.profile_path) if member.profile_path else None,
        )
        db.add(crew)
        await db.flush()
        db.add(MovieCastCrew(movie_id=movie.id, crew_person_id=crew.id))

    for prov in details.movie_providers:
        provider = await _get_or_create_provider(
            db, prov.name, prov.provider_type.value
        )
        exists = await db.execute(
            select(MovieProvider).where(
                MovieProvider.movie_id == movie.id,
                MovieProvider.provider_id == provider.id,
            )
        )
        if not exists.scalar_one_or_none():
            db.add(MovieProvider(movie_id=movie.id, provider_id=provider.id))

    await db.commit()
    return movie


def movie_to_details(movie: Movie) -> MovieDetails:
    """Convert a hydrated Movie ORM instance to a MovieDetails response."""
    return MovieDetails(
        movie_db_id=movie.id,
        tmdb_id=movie.tmdb_id,
        title=movie.title,
        poster_url=movie.poster_url or "",
        release_year=movie.release_year or 0,
        rating=movie.tmdb_rating or 0.0,
        genres=[g.name for g in movie.genres],
        is_adult=movie.is_adult or False,
        synopsis=movie.synopsis or "",
        runtime=movie.runtime or 0,
        trailer_url=movie.trailer_url,
        cast=[
            CastMember(
                name=cp.name,
                role_type=cp.role_type,
                profile_path=cp.image_url,
            )
            for cp in movie.cast_crew
        ],
        movie_providers=[
            MovieProviderSchema(
                name=p.name,
                provider_type=p.provider_type,
            )
            for p in movie.providers
        ],
    )


async def _get_or_create_genre(db: AsyncSession, name: str) -> Genre:
    result = await db.execute(select(Genre).where(Genre.name == name))
    genre = result.scalar_one_or_none()
    if not genre:
        genre = Genre(name=name)
        db.add(genre)
        await db.flush()
    return genre


async def _get_or_create_provider(
    db: AsyncSession, name: str, provider_type: str
) -> Provider:
    result = await db.execute(
        select(Provider).where(
            Provider.name == name, Provider.provider_type == provider_type
        )
    )
    provider = result.scalar_one_or_none()
    if not provider:
        provider = Provider(name=name, provider_type=provider_type)
        db.add(provider)
        await db.flush()
    return provider
