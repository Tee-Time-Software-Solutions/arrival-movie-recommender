from pathlib import Path
from typing import List, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.CRUD.movies import create_movie_stub
from movie_recommender.schemas.requests.interactions import SwipeAction


class Recommender:
    def __init__(self) -> None:
        pass

    def get_top_n_recommendations(
        self, user_id: int, n: int, list_of_filtered_movies: List[int]
    ) -> List[Tuple[int, str]]:
        """
        Given a user_id and number of movies to retrieve, returns a list of (movie_db_id, title) tuples.
        The returned movies respect filter list (movies to exclude).
        """
        # Mock data
        return [(1, "Arrival"), (2, "Interstellar"), (3, "The Matrix")]

    def get_similar_n_movies(self, movie_id: int, n: int):
        """Give a movie_id returns its closest movies. TODO: implement after MVP."""
        ...

    def set_user_feedback(
        self,
        user_id: int,
        movie_id: int,
        interaction_type: SwipeAction,
        is_supercharged: bool,
    ) -> None:
        """Belongs to online learning."""
        ...

    async def _write_movie(self, db: AsyncSession, title: str) -> int:
        """Write a movie stub (title only) to DB and return its auto-generated ID."""
        movie = await create_movie_stub(db, title)
        return movie.id

    def _ingest_csv_dataset(self, csv_path: Path):
        """
        Reads the CSV, filters for unique movie IDs,
        creates pairs of [(movie_id, movie_title)] then writes them to db.

        # TODO:
            1. Create object for ingest csv
            2. Processing CSV and write stub object
        """
        pass
