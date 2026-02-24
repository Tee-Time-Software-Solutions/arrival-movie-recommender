from functools import lru_cache

from movie_recommender.services.recommender.main import Recommender


@lru_cache(maxsize=1)
def get_recommender() -> Recommender:
    return Recommender()
