from typing import List, Tuple

from movie_recommender.schemas.users import UserPreferences


class Recommender:
    def __init__(self) -> None:
        pass

    def get_top_n(
        self, user_id: str, n: int, user_preferences: UserPreferences
    ) -> List[Tuple[int, str]]:
        """
        Def: given a user_id and number of movies to retrieve it returns a list of IDs of movies. These movie IDs must
             may be provided by the recommeder. They must be unique and not clash with existing ones in the db.
            The n returned  movies must respect user preferneces defined in the paremeter 'user_preferences'
        """
        # Mock data
        return [(1, "Arrival"), (2, "Interstellar"), (3, "The Matrix")]

    # missing update_user() and similar_movies()