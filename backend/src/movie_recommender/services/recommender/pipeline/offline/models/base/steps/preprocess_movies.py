import re

import pandas as pd

from movie_recommender.services.recommender.utils.schema import Config


def extract_year(title: str):
    match = re.search(r"\((\d{4})\)$", title)
    return int(match.group(1)) if match else None


def clean_title(title: str) -> str:
    return re.sub(r"\s*\(\d{4}\)$", "", title).strip()


def split_genres(genres: str) -> list[str]:
    if genres == "(no genres listed)":
        return []
    return genres.split("|")


def run(config: Config) -> None:
    source_path = config.data_dirs.source_dir / "movies.csv"
    processed_path = config.data_dirs.processed_dir / "movies_clean.parquet"

    print("Loading movies.csv...")
    df = pd.read_csv(
        source_path, dtype={"movieId": "int32", "title": "string", "genres": "string"}
    )
    df = df.rename(columns={"movieId": "movie_id"})
    df["release_year"] = df["title"].apply(extract_year)
    df["title"] = df["title"].apply(clean_title)
    df["genres"] = df["genres"].replace("(no genres listed)", "")
    df = df[["movie_id", "title", "release_year", "genres"]]

    df.to_parquet(processed_path, index=False)
    print(
        f"Movies done. Total: {len(df)}, missing years: {df['release_year'].isna().sum()}"
    )
