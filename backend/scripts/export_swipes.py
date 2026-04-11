#!/usr/bin/env python3
"""
Export Postgres `swipes` to recommender/data/raw/swipes_from_db.parquet.

Requires DB_* env vars (same as the API). Optional APP_USER_ID_OFFSET (default 10_000_000).

Usage (from backend/):
    uv run python scripts/export_swipes.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Allow `python scripts/export_swipes.py` from backend/ cwd
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT / "src"))

load_dotenv(_BACKEND_ROOT / ".env")

from movie_recommender.services.recommender.data_processing.swipe_export import (  # noqa: E402
    export_swipes_to_parquet,
    get_app_user_id_offset,
)
from movie_recommender.services.recommender.paths_dev import DATA_RAW  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DB_KEYS = (
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
    "DB_ASYNC_DRIVER",
)


def _database_url() -> str:
    missing = [k for k in _DB_KEYS if not os.getenv(k)]
    if missing:
        raise SystemExit(
            f"Missing required env for DB export: {', '.join(missing)}. "
            "Set them in .env or the environment."
        )
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    host = os.environ["DB_HOST"]
    port = os.environ["DB_PORT"]
    name = os.environ["DB_NAME"]
    driver = os.environ["DB_ASYNC_DRIVER"]
    return f"{driver}://{user}:{password}@{host}:{port}/{name}"


async def main_async(offset: int | None) -> None:
    engine = create_async_engine(_database_url(), echo=False)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as db:
        df = await export_swipes_to_parquet(db, app_user_id_offset=offset)
    await engine.dispose()
    used = offset if offset is not None else get_app_user_id_offset()
    logger.info("Exported %s swipes with APP_USER_ID_OFFSET=%s", len(df), used)
    logger.info("Wrote %s", DATA_RAW / "swipes_from_db.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export swipes to parquet for ALS pipeline")
    parser.add_argument(
        "--app-user-id-offset",
        type=int,
        default=None,
        help="Override APP_USER_ID_OFFSET for this run (MovieLens namespace separation)",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args.app_user_id_offset))


if __name__ == "__main__":
    main()
