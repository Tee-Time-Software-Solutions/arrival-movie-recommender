from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, text
from typing import List, Optional
import datetime


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    firebase_uid: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    profile_image_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    email: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP")
    )

    preferences: Mapped[Optional["Preference"]] = relationship(
        back_populates="user", uselist=False
    )


class Preference(Base):
    __tablename__ = "preferences"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), unique=True, nullable=False
    )
    min_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    max_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    min_rating: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP")
    )

    user: Mapped["User"] = relationship(back_populates="preferences")


class Genre(Base):
    __tablename__ = "genres"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)


class ExcludedGenre(Base):
    __tablename__ = "excluded_genres"

    genre_id: Mapped[int] = mapped_column(ForeignKey("genres.id"), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), primary_key=True)


class IncludedGenre(Base):
    __tablename__ = "included_genres"

    genre_id: Mapped[int] = mapped_column(ForeignKey("genres.id"), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), primary_key=True)


class Movie(Base):
    __tablename__ = "movies"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tmdb_id: Mapped[Optional[int]] = mapped_column(Integer, unique=True, nullable=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    poster_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    release_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tmdb_rating: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    synopsis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    runtime: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_adult: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    trailer_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP")
    )

    genres: Mapped[List["Genre"]] = relationship(
        secondary="movies_genres", viewonly=True
    )
    cast_crew: Mapped[List["CrewPerson"]] = relationship(
        secondary="movies_cast_crew", viewonly=True
    )
    providers: Mapped[List["Provider"]] = relationship(
        secondary="movies_providers", viewonly=True
    )


class MovieGenre(Base):
    __tablename__ = "movies_genres"

    movie_id: Mapped[int] = mapped_column(ForeignKey("movies.id"), primary_key=True)
    genre_id: Mapped[int] = mapped_column(ForeignKey("genres.id"), primary_key=True)


class CrewPerson(Base):
    __tablename__ = "crew_person"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    role_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    character_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    image_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)


class MovieCastCrew(Base):
    __tablename__ = "movies_cast_crew"

    movie_id: Mapped[int] = mapped_column(ForeignKey("movies.id"), primary_key=True)
    crew_person_id: Mapped[int] = mapped_column(
        ForeignKey("crew_person.id"), primary_key=True
    )


class Provider(Base):
    __tablename__ = "providers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    provider_type: Mapped[str] = mapped_column(String(32), nullable=False)


class MovieProvider(Base):
    __tablename__ = "movies_providers"

    movie_id: Mapped[int] = mapped_column(ForeignKey("movies.id"), primary_key=True)
    provider_id: Mapped[int] = mapped_column(
        ForeignKey("providers.id"), primary_key=True
    )


class Swipe(Base):
    __tablename__ = "swipes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    movie_id: Mapped[int] = mapped_column(ForeignKey("movies.id"), nullable=False)
    action_type: Mapped[str] = mapped_column(String(16), nullable=False)
    is_supercharged: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP")
    )

    user: Mapped["User"] = relationship("User")
    movie: Mapped["Movie"] = relationship("Movie")
