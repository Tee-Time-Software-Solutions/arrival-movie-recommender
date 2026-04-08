from pydantic import BaseModel


class WatchlistAddResponse(BaseModel):
    movie_id: int
    added: bool


class WatchlistRemoveResponse(BaseModel):
    movie_id: int
    removed: bool
