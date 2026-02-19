# recommender/training/preprocess_movies.py
import re
from pathlib import Path
import pandas as pd
from movie_recommender.services.recommender.paths_dev import DATA_RAW, DATA_PROCESSED

RAW_PATH = DATA_RAW / "movies.csv"
PROCESSED_PATH = DATA_PROCESSED / "movies_clean.parquet"


def extract_year(title: str):
    """
    Extract release year from movie title.
    Example: 'Toy Story (1995)' -> 1995
    """
    match = re.search(r"\((\d{4})\)$", title)
    if match:
        return int(match.group(1))
    return None


def clean_title(title: str):
    """
    Remove trailing year from title.
    Example: 'Toy Story (1995)' -> 'Toy Story'
    """
    return re.sub(r"\s*\(\d{4}\)$", "", title).strip()


def split_genres(genres: str):
    """
    Convert pipe-separated genres string to list.
    '(no genres listed)' -> []
    """
    if genres == "(no genres listed)":
        return []
    return genres.split("|")


def preprocess_movies():
    print("Loading movies.csv...")

    df = pd.read_csv(
        RAW_PATH,
        dtype={
            "movieId": "int32",
            "title": "string",
            "genres": "string",
        },
    )

    # Rename columns
    df = df.rename(columns={"movieId": "movie_id"})

    # Extract release year
    df["release_year"] = df["title"].apply(extract_year)

    # Clean title
    df["title"] = df["title"].apply(clean_title)

    # Split genres
    df["genres"] = df["genres"].apply(split_genres)

    # Reorder columns
    df = df[
        [
            "movie_id",
            "title",
            "release_year",
            "genres",
        ]
    ]

    # Create processed folder if needed
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save as parquet
    df.to_parquet(PROCESSED_PATH, index=False)

    # ---- Sanity checks ----
    print("Movies preprocessing complete.")
    print(f"Total movies: {len(df)}")
    print(f"Missing release years: {df['release_year'].isna().sum()}")
    print(f"Unique genres: {len(set(g for genres in df['genres'] for g in genres))}")


if __name__ == "__main__":
    preprocess_movies()
