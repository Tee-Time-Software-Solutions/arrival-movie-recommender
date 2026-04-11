"""
Export production swipes from Postgres to parquet for offline training.

App user ids are mapped to a disjoint namespace from MovieLens:
    ml_user_id = app_user_id + APP_USER_ID_OFFSET

Preference values match serving via swipe_to_preference (-2 .. +2).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
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
    return int(ts.timestamp())


def swipe_row_to_preference(action_type: str, is_supercharged: bool) -> int:
    action = SwipeAction(action_type)
    return swipe_to_preference(action, bool(is_supercharged))


def swipes_to_dataframe(
    rows: Sequence[Mapping[str, Any]],
    app_user_id_offset: int | None = None,
) -> pd.DataFrame:
    """
    Build a labeled export DataFrame from SQLAlchemy RowMappings or plain dicts.
    Each row: user_id, movie_id, action_type, is_supercharged, created_at.
    """
    off = (
        app_user_id_offset
        if app_user_id_offset is not None
        else get_app_user_id_offset()
    )
    records: list[dict[str, Any]] = []
    for row in rows:
        uid = int(row["user_id"])
        pref = swipe_row_to_preference(
            str(row["action_type"]), bool(row["is_supercharged"])
        )
        ts = row.get("created_at")
        records.append(
            {
                "app_user_id": uid,
                "user_id": ml_user_id_for_app_user(uid, off),
                "movie_id": int(row["movie_id"]),
                "action_type": str(row["action_type"]),
                "is_supercharged": bool(row["is_supercharged"]),
                "preference": pref,
                "timestamp": _timestamp_to_unix_seconds(ts),
            }
        )
    return pd.DataFrame.from_records(records)


async def fetch_all_swipe_rows(db: AsyncSession) -> list[Mapping[str, Any]]:
    result = await db.execute(
        select(
            swipes.c.user_id,
            swipes.c.movie_id,
            swipes.c.action_type,
            swipes.c.is_supercharged,
            swipes.c.created_at,
        ).order_by(swipes.c.created_at.asc())
    )
    # RowMappings are Mapping-like; avoid materializing a dict per row here.
    return result.mappings().all()


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
