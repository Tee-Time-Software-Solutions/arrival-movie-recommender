from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl
from enum import Enum


class MovieCard(BaseModel):
    movie_id: str = Field(..., description="Unique movie ID in the database.")
    tmdb_id: str = Field(..., description="Unique TMDB ID in the database.")
    title: str = Field(..., description="Movie title from the database.")
    poster_url: str = Field(
        ..., description="High-quality poster image URL for the movie card."
    )
    release_year: int = Field(..., description="Year the movie was released.")
    rating: float = Field(
        ...,
        description="TMDB-based rating out of 10. We are not using ratings from the recommendation DB",
        examples=7.5,
    )
    genres: List[str] = Field(
        ..., description="List of genres associated with the movie."
    )
    is_adult: bool = Field(..., description="Wether film has been rated 18+")


class CastMember(BaseModel):
    name: str = Field(..., example="Christopher Nolan")
    role_type: Optional[str] = Field(None, example="Director or Lead Actor")
    profile_path: Optional[HttpUrl] = Field(
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
    trailer_url: HttpUrl = Field(
        ..., description="Youtube or Vimeo link for the trailer [english]"
    )
    runtime: int = Field(..., description="Total time of the movie (in minutes)")
    movie_providers: List[MovieProvider] = Field(
        ...,
        description="List of providers offering this movie for streaming, rent, or buy.",
        example=[
            {"provider_type": "streaming", "name": "Netflix"},
            {"provider_type": "rent", "name": "Amazon Prime Video"},
        ],
    )
