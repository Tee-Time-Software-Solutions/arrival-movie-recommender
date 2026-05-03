"""Seed minimal data so eval_chatbot_vs_baseline.py and failure_analysis_chatbot.py
can run end-to-end without the full app being usable (offline-friendly: only
needs the MovieLens parquet + links.csv that are already on disk).

Inserts:
  - 100 movies (top of movies_filtered.parquet, with real tmdb_ids from links.csv)
  - genre rows + movies_genres links
  - 1 fake user (firebase_uid="eval-test-user")
  - 25 swipes for that user (15 likes, 5 dislikes, 5 skips), spread across
    different created_at values so leave-one-out picks a deterministic holdout.

Idempotent: if the user already exists, swipes/movies are upserted via INSERT
.. ON CONFLICT DO NOTHING (PG-specific).

Usage:
    cd backend && .venv/bin/python scripts/seed_eval_data.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
load_dotenv(ROOT / "env_config" / "synced" / ".env.dev", override=False)
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: E402

from movie_recommender.database.engine import DatabaseEngine  # noqa: E402
from movie_recommender.database.models import (  # noqa: E402
    genres as genres_table,
    movies as movies_table,
    movies_genres,
    swipes as swipes_table,
    users as users_table,
)


ARTIFACTS = ROOT / "src/movie_recommender/services/recommender/pipeline/artifacts"
MOVIES_PARQUET = ARTIFACTS / "dataset/processed/movies_filtered.parquet"
LINKS_CSV = ARTIFACTS / "dataset/source/small/links.csv"

N_MOVIES = 100
N_LIKES = 15
N_DISLIKES = 5
N_SKIPS = 5
TEST_FIREBASE_UID = "eval-test-user"
TEST_EMAIL = "eval-test-user@example.com"


def load_seed_movies() -> pd.DataFrame:
    movies_df = pd.read_parquet(MOVIES_PARQUET).head(N_MOVIES).copy()
    links_df = pd.read_csv(LINKS_CSV)
    links_df = links_df.rename(columns={"movieId": "movie_id"})
    df = movies_df.merge(links_df[["movie_id", "tmdbId"]], on="movie_id", how="left")
    df["tmdb_id"] = df["tmdbId"].where(df["tmdbId"].notna(), None)
    df["release_year"] = df["release_year"].where(df["release_year"].notna(), None)
    return df


async def upsert_movies(db, df: pd.DataFrame) -> list[int]:
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "id": int(r["movie_id"]),
            "tmdb_id": int(r["tmdb_id"]) if r["tmdb_id"] is not None else None,
            "title": str(r["title"])[:512],
            "release_year": int(r["release_year"]) if r["release_year"] is not None else None,
            "tmdb_rating": 7.0,
            "synopsis": "Seeded for eval — no TMDB hydration available offline.",
            "is_adult": False,
        })
    stmt = pg_insert(movies_table).values(rows).on_conflict_do_nothing(index_elements=["id"])
    await db.execute(stmt)
    return [r["id"] for r in rows]


async def upsert_genres_and_links(db, df: pd.DataFrame) -> None:
    all_genre_names: set[str] = set()
    for _, r in df.iterrows():
        for g in str(r["genres"]).split("|"):
            g = g.strip()
            if g:
                all_genre_names.add(g)

    if all_genre_names:
        await db.execute(
            pg_insert(genres_table)
            .values([{"name": g} for g in sorted(all_genre_names)])
            .on_conflict_do_nothing(index_elements=["name"])
        )

    result = await db.execute(select(genres_table.c.id, genres_table.c.name))
    name_to_id = {row.name: row.id for row in result}

    links: list[dict] = []
    for _, r in df.iterrows():
        for g in str(r["genres"]).split("|"):
            g = g.strip()
            if g and g in name_to_id:
                links.append({"movie_id": int(r["movie_id"]), "genre_id": name_to_id[g]})
    if links:
        await db.execute(
            pg_insert(movies_genres)
            .values(links)
            .on_conflict_do_nothing(index_elements=["movie_id", "genre_id"])
        )


async def upsert_user(db) -> int:
    result = await db.execute(
        select(users_table.c.id).where(users_table.c.firebase_uid == TEST_FIREBASE_UID)
    )
    row = result.first()
    if row:
        return row.id
    result = await db.execute(
        users_table.insert()
        .values(firebase_uid=TEST_FIREBASE_UID, email=TEST_EMAIL, onboarding_completed=True)
        .returning(users_table.c.id)
    )
    return result.scalar_one()


async def insert_swipes(db, user_id: int, movie_ids: list[int]) -> None:
    existing = await db.execute(
        select(swipes_table.c.movie_id).where(swipes_table.c.user_id == user_id)
    )
    already = {row.movie_id for row in existing}
    needed = [mid for mid in movie_ids if mid not in already]
    if not needed:
        print(f"  user {user_id} already has {len(already)} swipes — skipping seed.")
        return

    plan: list[tuple[str, list[int]]] = [
        ("like", needed[:N_LIKES]),
        ("dislike", needed[N_LIKES : N_LIKES + N_DISLIKES]),
        ("skip", needed[N_LIKES + N_DISLIKES : N_LIKES + N_DISLIKES + N_SKIPS]),
    ]
    base = datetime(2026, 4, 1, 12, 0, 0)
    rows = []
    delta = 0
    for action, ids in plan:
        for mid in ids:
            rows.append({
                "user_id": user_id,
                "movie_id": mid,
                "action_type": action,
                "is_supercharged": False,
                "created_at": base + timedelta(hours=delta),
            })
            delta += 1
    await db.execute(swipes_table.insert().values(rows))
    print(f"  inserted {len(rows)} swipes for user {user_id}.")


async def main() -> None:
    if not MOVIES_PARQUET.exists() or not LINKS_CSV.exists():
        print(f"Missing seed sources: {MOVIES_PARQUET} or {LINKS_CSV}")
        sys.exit(1)

    df = load_seed_movies()
    print(f"Seed sources loaded: {len(df)} movies.")

    session_factory = DatabaseEngine().session_factory
    async with session_factory() as db:
        movie_ids = await upsert_movies(db, df)
        await upsert_genres_and_links(db, df)
        user_id = await upsert_user(db)
        await insert_swipes(db, user_id, movie_ids)
        await db.commit()

    print(f"\nDone. Test user_id={user_id} ({TEST_FIREBASE_UID}).")
    print("Next: run scripts/eval_chatbot_vs_baseline.py and scripts/failure_analysis_chatbot.py")


if __name__ == "__main__":
    asyncio.run(main())
