from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional

from movie_recommender.schemas.requests.movies import MovieProvider
from datetime import datetime


class UserAnalytics(BaseModel):
    total_swipes: int
    total_likes: int
    total_dislikes: int
    total_seen: int
    top_genres: List[str]


class UserPreferences(BaseModel):
    included_genres: List[str] = Field(default_factory=list)
    excluded_genres: List[str] = Field(default_factory=list)
    min_release_year: Optional[int] = None
    max_release_year: Optional[int] = None
    min_rating: Optional[float] = None
    include_adult: bool = False
    movie_providers: List[MovieProvider] = Field(default_factory=list)


class UserDisplayInfo(BaseModel):
    username: str
    avatar_url: HttpUrl
    joined_at: str


class UserProfileSummary(BaseModel):
    profile: UserDisplayInfo
    stats: UserAnalytics
    preferences: UserPreferences


class UserCreate(BaseModel):
    firebase_uid: str
    profile_image_url: str
    email: str


class UserCreatedResponse(BaseModel):
    id: int
    firebase_uid: str
    profile_image_url: str
    email: str
    created_at: datetime
    updated_at: datetime
