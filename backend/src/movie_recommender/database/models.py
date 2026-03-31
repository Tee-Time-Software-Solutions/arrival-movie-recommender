from __future__ import annotations

from datetime import datetime
from typing import TypedDict

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    text,
)

metadata = MetaData()

# ── Tables ──────────────────────────────────────────────────────────

users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("firebase_uid", String(128), unique=True, nullable=False),
    Column("profile_image_url", String(512), nullable=True),
    Column("email", String(256), nullable=False),
    Column("created_at", DateTime, server_default=text("CURRENT_TIMESTAMP")),
    Column("updated_at", DateTime, server_default=text("CURRENT_TIMESTAMP")),
)

preferences = Table(
    "preferences",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id"), unique=True, nullable=False),
    Column("min_year", Integer, nullable=True),
    Column("max_year", Integer, nullable=True),
    Column("min_rating", Float, nullable=True),
    Column("include_adult", Boolean, nullable=True),
    Column("updated_at", DateTime, server_default=text("CURRENT_TIMESTAMP")),
)

genres = Table(
    "genres",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(64), unique=True, nullable=False),
)

excluded_genres = Table(
    "excluded_genres",
    metadata,
    Column("genre_id", Integer, ForeignKey("genres.id"), primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
)

included_genres = Table(
    "included_genres",
    metadata,
    Column("genre_id", Integer, ForeignKey("genres.id"), primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
)

movies = Table(
    "movies",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("tmdb_id", Integer, unique=True, nullable=True),
    Column("title", String(512), nullable=False),
    Column("poster_url", String(512), nullable=True),
    Column("release_year", Integer, nullable=True),
    Column("tmdb_rating", Float, nullable=True),
    Column("synopsis", Text, nullable=True),
    Column("runtime", Integer, nullable=True),
    Column("is_adult", Boolean, nullable=True),
    Column("trailer_url", String(512), nullable=True),
    Column("created_at", DateTime, server_default=text("CURRENT_TIMESTAMP")),
    Column("updated_at", DateTime, server_default=text("CURRENT_TIMESTAMP")),
)

movies_genres = Table(
    "movies_genres",
    metadata,
    Column("movie_id", Integer, ForeignKey("movies.id"), primary_key=True),
    Column("genre_id", Integer, ForeignKey("genres.id"), primary_key=True),
)

crew_person = Table(
    "crew_person",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("tmdb_person_id", Integer, unique=True, nullable=True),
    Column("name", String(256), nullable=False),
    Column("role_type", String(64), nullable=True),
    Column("character_name", String(256), nullable=True),
    Column("image_url", String(512), nullable=True),
)

movies_cast_crew = Table(
    "movies_cast_crew",
    metadata,
    Column("movie_id", Integer, ForeignKey("movies.id"), primary_key=True),
    Column("crew_person_id", Integer, ForeignKey("crew_person.id"), primary_key=True),
)

providers = Table(
    "providers",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(128), nullable=False),
    Column("provider_type", String(32), nullable=False),
)

movies_providers = Table(
    "movies_providers",
    metadata,
    Column("movie_id", Integer, ForeignKey("movies.id"), primary_key=True),
    Column("provider_id", Integer, ForeignKey("providers.id"), primary_key=True),
)

swipes = Table(
    "swipes",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id"), nullable=False),
    Column("movie_id", Integer, ForeignKey("movies.id"), nullable=False),
    Column("action_type", String(16), nullable=False),
    Column("is_supercharged", Boolean, default=False, nullable=False),
    Column("created_at", DateTime, server_default=text("CURRENT_TIMESTAMP")),
)

watchlist = Table(
    "watchlist",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id"), nullable=False),
    Column("movie_id", Integer, ForeignKey("movies.id"), nullable=False),
    Column("added_at", DateTime, server_default=text("CURRENT_TIMESTAMP")),
    UniqueConstraint("user_id", "movie_id", name="uq_watchlist_user_movie"),
)


# ── Row types (for IDE autocompletion on CRUD returns) ──────────────


class UserRow(TypedDict):
    id: int
    firebase_uid: str
    profile_image_url: str | None
    email: str
    created_at: datetime
    updated_at: datetime


class MovieRow(TypedDict):
    id: int
    tmdb_id: int | None
    title: str
    poster_url: str | None
    release_year: int | None
    tmdb_rating: float | None
    synopsis: str | None
    runtime: int | None
    is_adult: bool | None
    trailer_url: str | None
    created_at: datetime
    updated_at: datetime


class PreferenceRow(TypedDict):
    id: int
    user_id: int
    min_year: int | None
    max_year: int | None
    min_rating: float | None
    include_adult: bool | None
    updated_at: datetime


class SwipeRow(TypedDict):
    id: int
    user_id: int
    movie_id: int
    action_type: str
    is_supercharged: bool
    created_at: datetime


class WatchlistRow(TypedDict):
    id: int
    user_id: int
    movie_id: int
    added_at: datetime
