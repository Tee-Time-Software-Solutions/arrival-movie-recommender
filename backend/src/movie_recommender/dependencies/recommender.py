from movie_recommender.services.recommender.main import Recommender


_recommender_instance: Recommender | None = None


def get_recommender() -> Recommender:
    """
    Return a shared recommender instance so online user updates are preserved
    across requests in the current process.
    """
    global _recommender_instance

    if _recommender_instance is None:
        _recommender_instance = Recommender()

    return _recommender_instance
