"""Transform raw movies.csv into structured metadata."""

import re
from pathlib import Path

import pandas as pd

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[7]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "ml-20m" / "movies.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "movies_clean.parquet"


def main() -> None:
    # 1. Load with proper types
    df = pd.read_csv(
        RAW_PATH,
        dtype={
            "movieId": "int32",
            "title": "string",
            "genres": "string",
        },
    )

    # 2. Rename columns
    df = df.rename(columns={"movieId": "movie_id"})

    # 3. Extract release_year from title
    year_pattern = re.compile(r"\((\d{4})\)$")
    df["title"] = df["title"].astype(str)
    year_matches = df["title"].str.extract(year_pattern, expand=False)
    df["release_year"] = pd.to_numeric(year_matches, errors="coerce").astype("Int32")
    df["title"] = df["title"].str.replace(year_pattern, "", regex=True).str.strip()

    # 4. Split genres
    def split_genres(genres: str) -> list:
        if pd.isna(genres) or genres == "(no genres listed)":
            return []
        return [g.strip() for g in str(genres).split("|") if g.strip()]

    df["genres"] = df["genres"].apply(split_genres)

    # 5. Reorder columns
    df = df[["movie_id", "title", "genres", "release_year"]]

    # 6. Save output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    # 7. Sanity checks
    all_genres = set()
    for gs in df["genres"]:
        all_genres.update(gs)
    print(f"Total movies: {len(df):,}")
    print(f"Missing years count: {df['release_year'].isna().sum():,}")
    print(f"Number of unique genres: {len(all_genres)}")


if __name__ == "__main__":
    main()
