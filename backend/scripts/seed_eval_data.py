"""Seed realistic eval data: N app users sampled from MovieLens preferences.

Each seeded user is an app user whose swipes are copied from a real MovieLens
user's ratings (preference >=1 -> 'like', <=-1 -> 'dislike'). After running
this, retrain ALS WITHOUT `SKIP_DB_SWIPE_EXPORT=1` so the app swipes flow into
training and the new users get trained vectors at id = `app_user_id + 10_000_000`.

Idempotent: existing seed users (firebase_uid prefix `eval-ml-`) are not duplicated.

Usage:
    cd backend && .venv/bin/python scripts/seed_eval_data.py [--n-users 10]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
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
RATINGS_PARQUET = ARTIFACTS / "dataset/processed/ratings_clean.parquet"
MOVIES_PARQUET = ARTIFACTS / "dataset/processed/movies_filtered.parquet"
LINKS_CSV = ARTIFACTS / "dataset/source/small/links.csv"

MIN_LIKES = 15
MAX_LIKES = 60
SEED = 42
FIREBASE_PREFIX = "eval-ml-"


def pick_ml_users(ratings: pd.DataFrame, n: int) -> list[int]:
    """N MovieLens users whose like-count is in [MIN_LIKES, MAX_LIKES] — realistic, not power users."""
    likes = ratings[ratings.preference >= 1].groupby("user_id").size()
    eligible = likes[(likes >= MIN_LIKES) & (likes <= MAX_LIKES)].index.tolist()
    rng = random.Random(SEED)
    rng.shuffle(eligible)
    return eligible[:n]


def build_movie_rows(movie_ids: set[int]) -> list[dict]:
    movies = pd.read_parquet(MOVIES_PARQUET).set_index("movie_id")
    links = pd.read_csv(LINKS_CSV).rename(columns={"movieId": "movie_id"}).set_index("movie_id")
    rows = []
    for mid in sorted(movie_ids):
        if mid not in movies.index:
            continue
        m = movies.loc[mid]
        tmdb_id = None
        if mid in links.index:
            v = links.loc[mid, "tmdbId"]
            if pd.notna(v):
                tmdb_id = int(v)
        ry = m["release_year"]
        rows.append({
            "id": int(mid),
            "tmdb_id": tmdb_id,
            "title": str(m["title"])[:512],
            "release_year": int(ry) if pd.notna(ry) else None,
            "tmdb_rating": 7.0,
            "synopsis": "Seeded for eval — no TMDB hydration available offline.",
            "is_adult": False,
        })
    return rows


def build_genre_links(movie_ids: set[int]) -> tuple[set[str], list[tuple[int, str]]]:
    movies = pd.read_parquet(MOVIES_PARQUET).set_index("movie_id")
    all_genres: set[str] = set()
    pairs: list[tuple[int, str]] = []
    for mid in movie_ids:
        if mid not in movies.index:
            continue
        for g in str(movies.loc[mid, "genres"]).split("|"):
            g = g.strip()
            if g:
                all_genres.add(g)
                pairs.append((int(mid), g))
    return all_genres, pairs


async def upsert_movies(db, rows: list[dict]) -> None:
    if not rows:
        return
    BATCH = 500
    for i in range(0, len(rows), BATCH):
        await db.execute(
            pg_insert(movies_table).values(rows[i : i + BATCH])
            .on_conflict_do_nothing(index_elements=["id"])
        )


async def upsert_genres(db, names: set[str], pairs: list[tuple[int, str]]) -> None:
    if names:
        await db.execute(
            pg_insert(genres_table).values([{"name": g} for g in sorted(names)])
            .on_conflict_do_nothing(index_elements=["name"])
        )
    result = await db.execute(select(genres_table.c.id, genres_table.c.name))
    name_to_id = {row.name: row.id for row in result}
    link_rows = [
        {"movie_id": mid, "genre_id": name_to_id[g]}
        for mid, g in pairs if g in name_to_id
    ]
    if link_rows:
        BATCH = 1000
        for i in range(0, len(link_rows), BATCH):
            await db.execute(
                pg_insert(movies_genres).values(link_rows[i : i + BATCH])
                .on_conflict_do_nothing(index_elements=["movie_id", "genre_id"])
            )


async def upsert_user(db, ml_user_id: int) -> int:
    fb_uid = f"{FIREBASE_PREFIX}{ml_user_id}"
    result = await db.execute(select(users_table.c.id).where(users_table.c.firebase_uid == fb_uid))
    row = result.first()
    if row:
        return row.id
    result = await db.execute(
        users_table.insert()
        .values(firebase_uid=fb_uid, email=f"{fb_uid}@example.com", onboarding_completed=True)
        .returning(users_table.c.id)
    )
    return result.scalar_one()


async def insert_swipes(db, app_user_id: int, ml_ratings: pd.DataFrame) -> int:
    existing = await db.execute(
        select(swipes_table.c.movie_id).where(swipes_table.c.user_id == app_user_id)
    )
    seen = {r.movie_id for r in existing}

    base = datetime(2026, 4, 1, 12, 0, 0)
    rows = []
    for delta, r in enumerate(ml_ratings.itertuples()):
        if r.movie_id in seen:
            continue
        if r.preference >= 1:
            action = "like"
        elif r.preference <= -1:
            action = "dislike"
        else:
            continue
        rows.append({
            "user_id": app_user_id,
            "movie_id": int(r.movie_id),
            "action_type": action,
            "is_supercharged": False,
            "created_at": base + timedelta(minutes=delta),
        })
    if rows:
        await db.execute(swipes_table.insert().values(rows))
    return len(rows)


async def main(n_users: int) -> None:
    if not RATINGS_PARQUET.exists():
        print(f"Missing {RATINGS_PARQUET}")
        sys.exit(1)

    ratings = pd.read_parquet(RATINGS_PARQUET)
    valid_movie_ids = set(pd.read_parquet(MOVIES_PARQUET)["movie_id"].astype(int).tolist())
    ratings = ratings[ratings.movie_id.isin(valid_movie_ids)]

    ml_users = pick_ml_users(ratings, n_users)
    print(f"Picked {len(ml_users)} MovieLens users (each with {MIN_LIKES}–{MAX_LIKES} likes "
          f"on the {len(valid_movie_ids)}-movie filtered catalog).")

    user_ratings = {u: ratings[ratings.user_id == u] for u in ml_users}
    movie_ids: set[int] = set()
    for df in user_ratings.values():
        movie_ids.update(int(m) for m in df.movie_id.unique())
    print(f"Distinct movies referenced: {len(movie_ids)}")

    session_factory = DatabaseEngine().session_factory
    async with session_factory() as db:
        await upsert_movies(db, build_movie_rows(movie_ids))
        names, pairs = build_genre_links(movie_ids)
        await upsert_genres(db, names, pairs)
        total = 0
        for ml_uid in ml_users:
            app_uid = await upsert_user(db, ml_uid)
            n = await insert_swipes(db, app_uid, user_ratings[ml_uid])
            total += n
            print(f"  {FIREBASE_PREFIX}{ml_uid:>4} -> app_user_id={app_uid:>3}, +{n:>3} swipes")
        await db.commit()

    print(f"\nDone. {total} swipes inserted across {len(ml_users)} users.")
    print("Next steps:")
    print("  1. Retrain ALS so these users get trained vectors:")
    print("     cd backend && DB_HOST=localhost .venv/bin/python -m "
          "movie_recommender.services.recommender.pipeline.offline.models.als.main")
    print("  2. Run eval:")
    print("     .venv/bin/python scripts/eval_chatbot_vs_baseline.py --n-users 10 --k 10 "
          "--require-in-training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(main(args.n_users))
