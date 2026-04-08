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
    tmdb_person_id: Optional[int] = Field(
        None, description="TMDB person ID for cross-system linking"
    )


class ProviderType(str, Enum):
    FLATRATE = "flatrate"
    RENT = "rent"
    BUY = "buy"


class MovieProvider(BaseModel):
    name: str
    provider_type: ProviderType


class TMDBKeyword(BaseModel):
    tmdb_id: int
    name: str


class TMDBCollection(BaseModel):
    tmdb_id: int
    name: str
    part_number: Optional[int] = None


class TMDBProductionCompany(BaseModel):
    tmdb_id: int
    name: str
    origin_country: Optional[str] = None


class EntityReference(BaseModel):
    entity_type: str = Field(..., description="Person, Genre, Movie, Keyword, etc.")
    tmdb_id: int
    name: str


class ExplanationResponse(BaseModel):
    text: str = Field(
        ..., description="Human-readable explanation with @EntityName markers"
    )
    entities: List[EntityReference] = Field(
        default_factory=list, description="All referenced entities for frontend linking"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Explanation confidence score"
    )


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
    keywords: List[TMDBKeyword] = Field(
        default_factory=list, description="TMDB keywords for KG enrichment"
    )
    collection: Optional[TMDBCollection] = Field(
        None, description="Collection this movie belongs to (e.g. franchise)"
    )
    production_companies: List[TMDBProductionCompany] = Field(
        default_factory=list, description="Production companies for KG enrichment"
    )
    genre_tmdb_ids: List[int] = Field(
        default_factory=list, description="TMDB genre IDs for KG enrichment"
    )
    explanation: Optional[ExplanationResponse] = Field(
        None, description="KG-derived explanation for why this movie was recommended"
    )


class PaginatedMovieDetails(BaseModel):
    items: List[MovieDetails] = Field(..., description="Page of movie details.")
    total: int = Field(..., description="Total number of liked movies.")
    limit: int
    offset: int
