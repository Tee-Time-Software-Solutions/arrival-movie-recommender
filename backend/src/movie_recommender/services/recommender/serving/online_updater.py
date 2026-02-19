import numpy as np


def update_user_vector(
    user_vector: np.ndarray,
    movie_vector: np.ndarray,
    preference: int,
    eta: float = 0.05,
    norm_cap: float = 10.0,
) -> np.ndarray:
    """
    Apply one online preference update to a user embedding.

    preference:
    - negative for dislikes
    - positive for likes
    - zero for neutral/skip (no update)
    """
    if preference == 0:
        return user_vector.copy()

    updated = user_vector + (eta * preference * movie_vector)

    norm = float(np.linalg.norm(updated))
    if norm_cap > 0 and norm > norm_cap:
        updated = updated * (norm_cap / norm)

    return updated.astype(np.float32, copy=False)
