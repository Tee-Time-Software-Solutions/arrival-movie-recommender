from enum import Enum
from pydantic import BaseModel, Field

from movie_recommender.schemas.movies import MovieDetails


class SwipeAction(str, Enum):
    LIKE = "like"
    DISLIKE = "dislike"
    SKIP = "skip"


class SwipeRequest(BaseModel):
    action_type: SwipeAction
    is_supercharged: bool = False


class RegisteredFeedback(BaseModel):
    interaction_id: str
    movie_id: int
    action_type: SwipeAction
    is_supercharged: bool = False
    registered: bool


class RateRequest(BaseModel):
    rating: float = Field(..., ge=1, le=5)


class RatedMovie(BaseModel):
    movie: MovieDetails
    user_rating: float
    rated_at: str
