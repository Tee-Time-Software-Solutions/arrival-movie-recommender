from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class MovieCard(BaseModel):
    movie_db_id: int = Field(..., description="Unique movie ID in the database.")
    tmdb_id: int = Field(..., description="TMDB ID.")
    title: str = Field(..., description="Movie title.")
    poster_url: str = Field(..., description="Poster image URL.")
    release_year: int = Field(..., description="Year the movie was released.")
    rating: float = Field(..., description="TMDB rating out of 10.", examples=[7.5])
    genres: List[str] = Field(..., description="Genres associated with the movie.")
    is_adult: bool = Field(..., description="Whether film has been rated 18+")


class CastMember(BaseModel):
    name: str = Field(..., example="Amy Adams")
    role_type: str = Field(
        ..., example="Actor", description="Actor, Director, or Producer"
    )
    character_name: Optional[str] = Field(
        None,
        example="Louise Banks",
        description="Character played (only for actors, null for directors/producers)",
    )
    profile_path: Optional[str] = Field(
        None, description="Image url for the cast member"
    )


class ProviderType(str, Enum):
    FLATRATE = "flatrate"
    RENT = "rent"
    BUY = "buy"


class MovieProvider(BaseModel):
    name: str
    provider_type: ProviderType


class MovieDetails(MovieCard):
    synopsis: str
    cast: List[CastMember] = Field(default_factory=list)
    trailer_url: Optional[str] = Field(
        None, description="Youtube or Vimeo link for the trailer"
    )
    runtime: int = Field(..., description="Total time of the movie (in minutes)")
    movie_providers: List[MovieProvider] = Field(
        default_factory=list,
        description="Providers offering this movie for streaming, rent, or buy.",
    )
