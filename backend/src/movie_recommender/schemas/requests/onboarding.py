from typing import List, Optional

from pydantic import BaseModel, Field


class OnboardingMovieCard(BaseModel):
    movie_db_id: int
    tmdb_id: int
    title: str
    poster_url: str
    release_year: int
    tmdb_rating: float
    genres: List[str]


class OnboardingSearchResult(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_year: Optional[int] = None


class OnboardingSubmission(BaseModel):
    grid_movie_ids: List[int] = Field(..., min_length=5, max_length=30)
    search_movie_tmdb_ids: List[int] = Field(..., min_length=1, max_length=3)


class OnboardingCompleteResponse(BaseModel):
    onboarding_completed: bool
    movies_with_embeddings: int
