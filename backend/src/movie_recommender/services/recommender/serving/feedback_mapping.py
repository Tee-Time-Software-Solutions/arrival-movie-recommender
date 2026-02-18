from movie_recommender.schemas.interactions import SwipeAction


def swipe_to_preference(action_type: SwipeAction, is_supercharged: bool) -> int:
    """
    Contract:
    - DISLIKE -> -1, or -2 if superdislike
    - LIKE -> +1, or +2 if superlike
    (bundled the super into a supercharger that boosts the like/dislike, rather than having each)
    """
    if action_type == SwipeAction.SKIP:
        return 0

    if action_type == SwipeAction.DISLIKE:
        if is_supercharged:
            return -2
        else:
            return -1

    if action_type == SwipeAction.LIKE:
        if is_supercharged:
            return 2
        else:
            return 1

    raise ValueError(f"Unsupported swipe action: {action_type}")
