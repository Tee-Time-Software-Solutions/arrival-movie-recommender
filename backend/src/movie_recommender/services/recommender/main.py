from pathlib import Path
from typing import List, Tuple

from movie_recommender.schemas.interactions import SwipeAction
from movie_recommender.schemas.users import UserPreferences


class Recommender:
    def __init__(self) -> None:
        pass

    def get_top_n_recommendations(
        self, user_id: str, n: int, list_of_filtered_movies: List[int]
    ) -> List[Tuple[int, str]]:
        """
        Def: given a user_id and number of movies to retrieve it returns a list of IDs of movies. These movie IDs must
             may be provided by the recommeder. They must be unique and not clash with existing ones in the db.
            The n returned  movies must respect user preferneces defined in the paremeter 'user_preferences'

        TODO:
            change user_preferences param to 'list_of_filtered_movies: List[ids:int]'
        """
        # Mock data
        return [(1, "Arrival"), (2, "Interstellar"), (3, "The Matrix")]

    def get_similar_n_movies(self, movie_id: int, n: int):
        """
        Def: Give a movie_id returns its closes movies

        TODO:
            - Implement after MVP
        """

    def set_user_feedback(
        self,
        user_id: str,
        movie_id: int,
        interaction_type: SwipeAction,
        is_supercharged: bool,
    ) -> None:
        """
        Belongs to:
            - Online learning
        """
        ...

    def _ingest_csv_dataset(self, csv_path: Path):
        """
        De:
            Reads the CSV, filters for unique movie IDs, createes pairs of [(movie_id, movie_title)] then writes them to db
        Belongs to:
            - Offline learning
        """


# Add new function that id, movie_name out of ingested csv
