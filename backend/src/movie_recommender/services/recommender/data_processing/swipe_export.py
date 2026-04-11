"""
Export production swipes from Postgres to parquet for offline training.

App user ids are mapped to a disjoint namespace from MovieLens:
    ml_user_id = app_user_id + APP_USER_ID_OFFSET

Preference values match serving via swipe_to_preference (-2 .. +2).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.models import swipes
from movie_recommender.schemas.requests.interactions import SwipeAction
from movie_recommender.services.recommender.paths_dev import DATA_RAW
from movie_recommender.services.recommender.serving.feedback_mapping import (
    swipe_to_preference,
)

DEFAULT_APP_USER_ID_OFFSET = 10_000_000
ENV_APP_USER_ID_OFFSET = "APP_USER_ID_OFFSET"

SWIPES_FROM_DB_FILENAME = "swipes_from_db.parquet"


def get_app_user_id_offset() -> int:
    raw = os.getenv(ENV_APP_USER_ID_OFFSET, "").strip()
    if not raw:
        return DEFAULT_APP_USER_ID_OFFSET
    return int(raw)


def ml_user_id_for_app_user(app_user_id: int, offset: int | None = None) -> int:
    off = offset if offset is not None else get_app_user_id_offset()
    return int(app_user_id) + off


def _timestamp_to_unix_seconds(ts: datetime | None) -> int:
    if ts is None:
        return 0
    if ts.tzinfo is not None:
        return int(ts.timestamp())
    return int(ts.timestamp()) if hasattr(ts, "timestamp") else 0


def swipe_row_to_preference(action_type: str, is_supercharged: bool) -> int:
    action = SwipeAction(action_type)
    return swipe_to_preference(action, bool(is_supercharged))


def swipes_to_dataframe(
    rows: list[Any],
    app_user_id_offset: int | None = None,
) -> pd.DataFrame:
    """
    Build a labeled export DataFrame from SQLAlchemy row mappings (or dicts).
    Each row: user_id, movie_id, action_type, is_supercharged, created_at.
    """
    off = (
        app_user_id_offset
        if app_user_id_offset is not None
        else get_app_user_id_offset()
    )
    records: list[dict[str, Any]] = []
    for row in rows:
        r = dict(row) if not isinstance(row, dict) else row
        uid = int(r["user_id"])
        pref = swipe_row_to_preference(
            str(r["action_type"]), bool(r["is_supercharged"])
        )
        ts = r.get("created_at")
        records.append(
            {
                "app_user_id": uid,
                "user_id": ml_user_id_for_app_user(uid, off),
                "movie_id": int(r["movie_id"]),
                "action_type": str(r["action_type"]),
                "is_supercharged": bool(r["is_supercharged"]),
                "preference": pref,
                "timestamp": _timestamp_to_unix_seconds(ts),
            }
        )
    return pd.DataFrame.from_records(records)


async def fetch_all_swipe_rows(db: AsyncSession) -> list[dict[str, Any]]:
    result = await db.execute(
        select(
            swipes.c.user_id,
            swipes.c.movie_id,
            swipes.c.action_type,
            swipes.c.is_supercharged,
            swipes.c.created_at,
        ).order_by(swipes.c.created_at.asc())
    )
    return [dict(m) for m in result.mappings().all()]


async def export_swipes_to_parquet(
    db: AsyncSession,
    output_path: Any | None = None,
    app_user_id_offset: int | None = None,
) -> pd.DataFrame:
    """
    Read all swipes, write swipes_from_db.parquet under DATA_RAW, return DataFrame.
    """
    off = (
        app_user_id_offset
        if app_user_id_offset is not None
        else get_app_user_id_offset()
    )
    rows = await fetch_all_swipe_rows(db)
    df = swipes_to_dataframe(rows, off)
    path = output_path if output_path is not None else DATA_RAW / SWIPES_FROM_DB_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
