"""Write ``raw/swipes_from_db.parquet`` from Postgres (latest app interactions).

Runs before ``merge_interactions`` so the closed loop is a single pipeline entrypoint.
Skip when ``SKIP_DB_SWIPE_EXPORT=1`` (e.g. CI or MovieLens-only retrain).

APP_USER_ID_OFFSET (default 10_000_000) is added to every app user_id.
MovieLens user IDs go up to ~162 541, so without the offset app users would
collide with existing MovieLens rows in the training matrix.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime

import pandas as pd

from movie_recommender.core.settings.main import AppSettings
from movie_recommender.database.CRUD.interactions import get_all_swipes
from movie_recommender.services.recommender.utils.schema import Config, load_config

_SWIPES_FILENAME = "swipes_from_db.parquet"
_DEFAULT_OFFSET = 10_000_000

_PREFERENCE_MAP: dict[tuple[str, bool], int] = {
    ("like", False): 1,
    ("like", True): 2,
    ("dislike", False): -1,
    ("dislike", True): -2,
    ("skip", False): 0,
    ("skip", True): 0,
}


def get_app_user_id_offset() -> int:
    return int(os.environ.get("APP_USER_ID_OFFSET", _DEFAULT_OFFSET))


def ml_user_id_for_app_user(app_user_id: int, offset: int | None = None) -> int:
    if offset is None:
        offset = get_app_user_id_offset()
    return app_user_id + offset


def swipe_row_to_preference(action_type: str, is_supercharged: bool) -> int:
    return _PREFERENCE_MAP.get((action_type, bool(is_supercharged)), 0)


def swipes_to_dataframe(rows: list[dict], app_user_id_offset: int) -> pd.DataFrame:
    records = [
        {
            "app_user_id": row["user_id"],
            "user_id": ml_user_id_for_app_user(row["user_id"], app_user_id_offset),
            "movie_id": row["movie_id"],
            "preference": swipe_row_to_preference(
                row["action_type"], bool(row.get("is_supercharged"))
            ),
            "timestamp": (
                int(row["created_at"].timestamp())
                if isinstance(row.get("created_at"), datetime)
                else 0
            ),
        }
        for row in rows
    ]
    return pd.DataFrame(
        records,
        columns=["app_user_id", "user_id", "movie_id", "preference", "timestamp"],
    )


def run(config: Config) -> None:
    if os.environ.get("SKIP_DB_SWIPE_EXPORT", "").strip() in ("1", "true", "yes"):
        print("  SKIP_DB_SWIPE_EXPORT set — not fetching swipes from Postgres.")
        return

    output_path = config.data_dirs.source_dir.parent / "raw" / _SWIPES_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)

    app_user_id_offset = get_app_user_id_offset()

    async def _export() -> None:
        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )

        db_settings = AppSettings().database
        url = (
            f"{db_settings.async_driver}://{db_settings.user}:{db_settings.password}"
            f"@{db_settings.host}:{db_settings.port}/{db_settings.database}"
        )
        engine = create_async_engine(url, pool_size=2, max_overflow=0)
        session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        try:
            async with session_factory() as db:
                rows = await get_all_swipes(db)
        finally:
            await engine.dispose()

        df = swipes_to_dataframe(rows, app_user_id_offset)
        df.to_parquet(output_path, index=False)
        print(
            f"  Fetched {len(df):,} swipe rows → {output_path.name} "
            f"(APP_USER_ID_OFFSET={app_user_id_offset})"
        )

    asyncio.run(_export())


if __name__ == "__main__":
    run(load_config())
