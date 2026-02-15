from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional

from movie_recommender.schemas.movies import MovieProvider


class UserAnalytics(BaseModel):
    total_swipes: int
    total_likes: int
    total_dislikes: int
    top_genres: List[str]


class UserPreferences(BaseModel):
    preferred_genres: List[str]
    min_release_year: int
    include_adult: bool
    movie_providers: List[MovieProvider]


class UserDisplayInfo(BaseModel):
    username: str
    avatar_url: HttpUrl
    joined_at: str


class UserProfileSummary(BaseModel):
    profile: UserDisplayInfo
    stats: UserAnalytics
    preferences: UserPreferences
