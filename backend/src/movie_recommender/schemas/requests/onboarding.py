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
    movie_db_id: Optional[int] = None  # set if the movie is already in the DB
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_year: Optional[int] = None


class OnboardingSubmission(BaseModel):
    movie_db_ids: List[int] = Field(..., min_length=5, max_length=33)


class OnboardingCompleteResponse(BaseModel):
    onboarding_completed: bool
