"""
Curated seed movies for onboarding.

~50 universally recognized movies, 5 per genre.
ALL movies MUST exist in the MovieLens-25M training set (have embeddings).
Keys are genre names; values are TMDB IDs.
"""

import random

SEED_MOVIES: dict[str, list[int]] = {
    "Action": [
        155,     # The Dark Knight
        603,     # The Matrix
        98,      # Gladiator
        1891,    # The Empire Strikes Back
        101,     # Leon: The Professional
    ],
    "Adventure": [
        85,      # Raiders of the Lost Ark
        120,     # The Lord of the Rings: The Fellowship of the Ring
        11,      # Star Wars: A New Hope
        105,     # Back to the Future
        8587,    # The Lion King
    ],
    "Animation": [
        862,     # Toy Story
        129,     # Spirited Away
        12,      # Finding Nemo
        128,     # Princess Mononoke
        4935,    # Howl's Moving Castle
    ],
    "Comedy": [
        115,     # The Big Lebowski
        637,     # Life Is Beautiful
        37165,   # The Truman Show
        935,     # Dr. Strangelove
        100,     # Lock, Stock and Two Smoking Barrels
    ],
    "Crime": [
        238,     # The Godfather
        278,     # The Shawshank Redemption
        680,     # Pulp Fiction
        550,     # Fight Club
        769,     # GoodFellas
    ],
    "Drama": [
        13,      # Forrest Gump
        424,     # Schindler's List
        389,     # 12 Angry Men
        497,     # The Green Mile
        770,     # Gone with the Wind
    ],
    "Horror": [
        694,     # The Shining
        539,     # Psycho
        348,     # Alien
        1091,    # The Thing
        9552,    # The Exorcist
    ],
    "Romance": [
        597,     # Titanic
        289,     # Casablanca
        38,      # Eternal Sunshine of the Spotless Mind
        194,     # Amelie
        76,      # Before Sunrise
    ],
    "Sci-Fi": [
        185,     # A Clockwork Orange
        280,     # Terminator 2: Judgment Day
        19,      # Metropolis
        601,     # E.T. the Extra-Terrestrial
        78,      # Blade Runner
    ],
    "Thriller": [
        274,     # The Silence of the Lambs
        807,     # Se7en
        745,     # The Sixth Sense
        1124,    # The Prestige
        77,      # Memento
    ],
}

# Flattened set of all unique TMDB IDs for quick lookups
ALL_SEED_TMDB_IDS: set[int] = {
    tmdb_id for ids in SEED_MOVIES.values() for tmdb_id in ids
}


def sample_onboarding_movies(per_genre: int = 3) -> list[tuple[str, int]]:
    """Return a genre-balanced sample of (genre, tmdb_id) pairs.

    Picks `per_genre` movies from each genre (randomly), giving ~30 movies total.
    """
    sampled: list[tuple[str, int]] = []
    for genre, tmdb_ids in SEED_MOVIES.items():
        n = min(per_genre, len(tmdb_ids))
        for tmdb_id in random.sample(tmdb_ids, n):
            sampled.append((genre, tmdb_id))
    return sampled
