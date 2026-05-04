"""One-shot: rehydrate movies from TMDB and write them to Postgres + Neo4j.

Why this exists: `seed_eval_data.py` only seeds bare movie rows + genre links —
no cast/crew/keywords. As a result, even after `upsert_movie_to_kg`, the KG
contains Movie nodes with no relationships, so beacons stay empty during eval.

This script walks every movie row in Postgres that has a tmdb_id, fetches the
full TMDB detail, calls `save_hydrated_movie` (writes cast/crew/keywords to
Postgres) and `upsert_movie_to_kg` (writes to Neo4j). Idempotent.

Usage:
    cd backend
    set -a; source env_config/synced/.env.dev; set +a
    DB_HOST=localhost REDIS_URL=redis://localhost:6379 NEO4J_URI=bolt://localhost:7687 \\
        .venv/bin/python scripts/backfill_kg_from_db.py [--limit 100] [--user-swipes-only]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / "env_config" / "synced" / ".env.dev")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import select  # noqa: E402

from movie_recommender.core.clients.neo4j import Neo4jClient  # noqa: E402
from movie_recommender.database.CRUD.movies import save_hydrated_movie  # noqa: E402
from movie_recommender.database.engine import DatabaseEngine  # noqa: E402
from movie_recommender.database.models import movies, swipes  # noqa: E402
from movie_recommender.services.knowledge_graph.writer import (  # noqa: E402
    upsert_movie_to_kg,
)
from movie_recommender.services.recommender.pipeline.hydrator.main import (  # noqa: E402
    TMDBFetcher,
)


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


async def fetch_target_rows(
    session_factory, user_swipes_only: bool
) -> list[tuple[int, int]]:
    async with session_factory() as db:
        if user_swipes_only:
            stmt = (
                select(movies.c.id, movies.c.tmdb_id)
                .join(swipes, swipes.c.movie_id == movies.c.id)
                .where(movies.c.tmdb_id.isnot(None))
                .distinct()
            )
        else:
            stmt = select(movies.c.id, movies.c.tmdb_id).where(
                movies.c.tmdb_id.isnot(None)
            )
        rows = await db.execute(stmt)
        return [(r.id, r.tmdb_id) for r in rows.fetchall()]


async def main(limit: int | None, user_swipes_only: bool) -> None:
    session_factory = DatabaseEngine().session_factory
    driver = await Neo4jClient().get_async_driver()
    fetcher = TMDBFetcher()

    targets = await fetch_target_rows(session_factory, user_swipes_only)
    if limit:
        targets = targets[:limit]
    logger.info(
        "Backfilling KG from %d movies (user_swipes_only=%s)…",
        len(targets),
        user_swipes_only,
    )

    ok = 0
    skipped = 0
    failed = 0
    for i, (movie_db_id, tmdb_id) in enumerate(targets, start=1):
        try:
            detail = await fetcher.fetch_detail_by_id(int(tmdb_id))
            if not detail:
                skipped += 1
                continue
            details = fetcher.parse_detail_response(movie_db_id, detail)

            async with session_factory() as db:
                try:
                    await save_hydrated_movie(db, movie_db_id, details)
                except Exception as e:
                    logger.debug("save_hydrated_movie %s: %r", movie_db_id, e)

            await upsert_movie_to_kg(driver, details)
            ok += 1
        except Exception as e:
            failed += 1
            logger.warning("  movie_id=%s tmdb_id=%s: %r", movie_db_id, tmdb_id, e)

        if i % 25 == 0:
            logger.info(
                "  progress: %d/%d  (ok=%d skipped=%d failed=%d)",
                i,
                len(targets),
                ok,
                skipped,
                failed,
            )

    logger.info("Done. ok=%d  skipped=%d  failed=%d", ok, skipped, failed)
    await Neo4jClient().close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--user-swipes-only",
        action="store_true",
        help="Only backfill movies that any user has swiped on (faster, focused on eval).",
    )
    args = parser.parse_args()
    asyncio.run(main(args.limit, args.user_swipes_only))
