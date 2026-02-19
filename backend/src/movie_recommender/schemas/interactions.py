from enum import Enum
from pydantic import BaseModel


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
