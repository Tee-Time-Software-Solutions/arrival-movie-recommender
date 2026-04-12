"""
Curated seed movies for onboarding.

~50 universally recognized movies, 5 per genre.
ALL movies exist in the MovieLens-25M dataset.
Keys are genre names; values are (movielens_id, title) pairs.
MovieLens IDs verified against ml-latest-small/movies.csv.
"""

import random

# (movielens_id, title_as_stored_in_movielens)
SEED_MOVIES: dict[str, list[tuple[int, str]]] = {
    "Action": [
        (58559, "Dark Knight, The (2008)"),
        (2571, "Matrix, The (1999)"),
        (3578, "Gladiator (2000)"),
        (1196, "Star Wars: Episode V - The Empire Strikes Back (1980)"),
        (293, "Léon: The Professional (a.k.a. The Professional) (Léon) (1994)"),
    ],
    "Adventure": [
        (
            1198,
            "Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)",
        ),
        (4993, "Lord of the Rings: The Fellowship of the Ring, The (2001)"),
        (260, "Star Wars: Episode IV - A New Hope (1977)"),
        (1270, "Back to the Future (1985)"),
        (364, "Lion King, The (1994)"),
    ],
    "Animation": [
        (1, "Toy Story (1995)"),
        (5618, "Spirited Away (Sen to Chihiro no kamikakushi) (2001)"),
        (6377, "Finding Nemo (2003)"),
        (3000, "Princess Mononoke (Mononoke-hime) (1997)"),
        (31658, "Howl's Moving Castle (Hauru no ugoku shiro) (2004)"),
    ],
    "Comedy": [
        (1732, "Big Lebowski, The (1998)"),
        (2324, "Life Is Beautiful (La Vita è bella) (1997)"),
        (1682, "Truman Show, The (1998)"),
        (
            750,
            "Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)",
        ),
        (2542, "Lock, Stock & Two Smoking Barrels (1998)"),
    ],
    "Crime": [
        (858, "Godfather, The (1972)"),
        (318, "Shawshank Redemption, The (1994)"),
        (296, "Pulp Fiction (1994)"),
        (2959, "Fight Club (1999)"),
        (1213, "Goodfellas (1990)"),
    ],
    "Drama": [
        (356, "Forrest Gump (1994)"),
        (527, "Schindler's List (1993)"),
        (1203, "12 Angry Men (1957)"),
        (3147, "Green Mile, The (1999)"),
        (920, "Gone with the Wind (1939)"),
    ],
    "Horror": [
        (1258, "Shining, The (1980)"),
        (1219, "Psycho (1960)"),
        (1214, "Alien (1979)"),
        (2288, "Thing, The (1982)"),
        (1997, "Exorcist, The (1973)"),
    ],
    "Romance": [
        (1721, "Titanic (1997)"),
        (912, "Casablanca (1942)"),
        (7361, "Eternal Sunshine of the Spotless Mind (2004)"),
        (4973, "Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)"),
        (215, "Before Sunrise (1995)"),
    ],
    "Sci-Fi": [
        (1206, "Clockwork Orange, A (1971)"),
        (589, "Terminator 2: Judgment Day (1991)"),
        (2010, "Metropolis (1927)"),
        (1097, "E.T. the Extra-Terrestrial (1982)"),
        (541, "Blade Runner (1982)"),
    ],
    "Thriller": [
        (593, "Silence of the Lambs, The (1991)"),
        (47, "Seven (a.k.a. Se7en) (1995)"),
        (2762, "Sixth Sense, The (1999)"),
        (48780, "Prestige, The (2006)"),
        (4226, "Memento (2000)"),
    ],
}

ALL_SEED_ML_IDS: set[int] = {
    ml_id for entries in SEED_MOVIES.values() for ml_id, _ in entries
}


def sample_onboarding_movies(per_genre: int = 3) -> list[tuple[str, int, str]]:
    """Return a genre-balanced sample of (genre, ml_id, title) triples."""
    sampled: list[tuple[str, int, str]] = []
    for genre, entries in SEED_MOVIES.items():
        n = min(per_genre, len(entries))
        for ml_id, title in random.sample(entries, n):
            sampled.append((genre, ml_id, title))
    return sampled
