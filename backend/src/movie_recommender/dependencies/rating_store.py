from datetime import datetime, timezone
from typing import Dict, List

from movie_recommender.schemas.interactions import RatedMovie
from movie_recommender.schemas.movies import MovieDetails


class RatingStore:
    def __init__(self) -> None:
        self._ratings: Dict[str, List[RatedMovie]] = {}

    def add_rating(
        self, user_id: str, movie_id: str, rating: float, movie_details: MovieDetails
    ) -> None:
        if user_id not in self._ratings:
            self._ratings[user_id] = []

        # Upsert: remove existing rating for same movie
        self._ratings[user_id] = [
            r for r in self._ratings[user_id] if r.movie.movie_id != movie_id
        ]

        self._ratings[user_id].append(
            RatedMovie(
                movie=movie_details,
                user_rating=rating,
                rated_at=datetime.now(timezone.utc).isoformat(),
            )
        )

    def get_top_rated(self, user_id: str, limit: int = 10) -> List[RatedMovie]:
        ratings = self._ratings.get(user_id, [])
        return sorted(ratings, key=lambda r: r.user_rating, reverse=True)[:limit]


_rating_store_instance: RatingStore | None = None


def get_rating_store() -> RatingStore:
    global _rating_store_instance
    if _rating_store_instance is None:
        _rating_store_instance = RatingStore()
    return _rating_store_instance
