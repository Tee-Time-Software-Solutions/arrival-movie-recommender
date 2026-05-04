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
    links_path = config.data_dirs.source_dir / "links.csv"
    processed_path = config.data_dirs.processed_dir / "movies_clean.parquet"

    print("Loading movies.csv...")
    df = pd.read_csv(
        source_path, dtype={"movieId": "int32", "title": "string", "genres": "string"}
    )
    df = df.rename(columns={"movieId": "movie_id"})
    df["release_year"] = df["title"].apply(extract_year)
    df["title"] = df["title"].apply(clean_title)
    df["genres"] = df["genres"].replace("(no genres listed)", "")

    # Carries tmdb_id forward so the online recommender can talk to the Neo4j KG
    # (which keys nodes by tmdb_id).
    if links_path.exists():
        links = pd.read_csv(links_path, usecols=["movieId", "tmdbId"])
        links = links.rename(columns={"movieId": "movie_id", "tmdbId": "tmdb_id"})
        df = df.merge(links, on="movie_id", how="left")
        df["tmdb_id"] = df["tmdb_id"].astype("Int64")
    else:
        df["tmdb_id"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    df = df[["movie_id", "title", "release_year", "genres", "tmdb_id"]]

    df.to_parquet(processed_path, index=False)
    print(
        f"Movies done. Total: {len(df)}, missing years: {df['release_year'].isna().sum()}, "
        f"missing tmdb_id: {df['tmdb_id'].isna().sum()}"
    )
