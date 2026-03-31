"""
Beacon Map — weighted entity profile derived from a user's swipe history.

The beacon map identifies which entities (directors, actors, genres, keywords)
a user cares about, and how much. This is used during KG traversal to find
meaningful explanation paths.
"""

import json
import logging
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

from neo4j import AsyncDriver
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.models import swipes, movies

logger = logging.getLogger(__name__)

BEACON_REDIS_PREFIX = "beacon:user:"
BEACON_TTL_SECONDS = 86400  # 24 hours

# Entity-type weight multipliers
ENTITY_MULTIPLIERS = {
    "Director": 1.5,
    "Actor": 1.0,
    "Genre": 0.8,
    "Writer": 0.7,
    "Keyword": 0.5,
}

# Swipe action to preference score
SWIPE_SCORES = {
    ("like", True): 2.0,
    ("like", False): 1.0,
    ("dislike", True): -2.0,
    ("dislike", False): -1.0,
    ("skip", True): 0.0,
    ("skip", False): 0.0,
}

# Recency decay factor per day
RECENCY_DECAY = 0.95


@dataclass
class BeaconEntry:
    entity_type: str  # "Person", "Genre", "Keyword"
    tmdb_id: int
    name: str
    weight: float


BeaconMap = dict[tuple[str, int], BeaconEntry]


async def build_beacon_map(
    neo4j_driver: AsyncDriver, db: AsyncSession, user_id: int
) -> BeaconMap:
    """
    Full rebuild of the beacon map from the user's swipe history.
    Queries PostgreSQL for swipes, then Neo4j for entity connections.
    """
    result = await db.execute(
        select(
            swipes.c.movie_id,
            swipes.c.action_type,
            swipes.c.is_supercharged,
            swipes.c.created_at,
            movies.c.tmdb_id,
        )
        .join(movies, movies.c.id == swipes.c.movie_id)
        .where(swipes.c.user_id == user_id, movies.c.tmdb_id.isnot(None))
        .order_by(swipes.c.created_at.desc())
    )
    swipe_rows = result.fetchall()

    if not swipe_rows:
        return {}

    beacon_map: BeaconMap = {}
    now = datetime.now(timezone.utc)

    for row in swipe_rows:
        score = SWIPE_SCORES.get((row.action_type, row.is_supercharged), 0.0)
        if score == 0.0:
            continue

        # Recency decay
        days_ago = (now - row.created_at.replace(tzinfo=timezone.utc)).days
        decay = math.pow(RECENCY_DECAY, days_ago)
        weighted_score = score * decay

        # Get entities connected to this movie in the KG
        entities = await _get_movie_entities(neo4j_driver, row.tmdb_id)
        for entity_type, tmdb_id, name in entities:
            multiplier = ENTITY_MULTIPLIERS.get(entity_type, 0.5)
            key = (entity_type, tmdb_id)
            if key not in beacon_map:
                beacon_map[key] = BeaconEntry(
                    entity_type=entity_type,
                    tmdb_id=tmdb_id,
                    name=name,
                    weight=0.0,
                )
            beacon_map[key].weight += weighted_score * multiplier

    return beacon_map


async def _get_movie_entities(
    neo4j_driver: AsyncDriver, movie_tmdb_id: int
) -> list[tuple[str, int, str]]:
    """Get all entities connected to a movie in the KG."""
    entities = []
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (m:Movie {tmdb_id: $tmdb_id})
            OPTIONAL MATCH (m)-[:DIRECTED_BY]->(d:Person)
            OPTIONAL MATCH (a:Person)-[:ACTED_IN]->(m)
            OPTIONAL MATCH (m)-[:HAS_GENRE]->(g:Genre)
            OPTIONAL MATCH (m)-[:WRITTEN_BY]->(w:Person)
            OPTIONAL MATCH (m)-[:HAS_KEYWORD]->(k:Keyword)
            RETURN
                collect(DISTINCT {type: 'Director', id: d.tmdb_id, name: d.name}) AS directors,
                collect(DISTINCT {type: 'Actor', id: a.tmdb_id, name: a.name}) AS actors,
                collect(DISTINCT {type: 'Genre', id: g.tmdb_id, name: g.name}) AS genres,
                collect(DISTINCT {type: 'Writer', id: w.tmdb_id, name: w.name}) AS writers,
                collect(DISTINCT {type: 'Keyword', id: k.tmdb_id, name: k.name}) AS keywords
            """,
            tmdb_id=movie_tmdb_id,
        )
        record = await result.single()
        if not record:
            return entities

        for group_key in ["directors", "actors", "genres", "writers", "keywords"]:
            for item in record[group_key]:
                if item["id"] is not None:
                    entities.append((item["type"], item["id"], item["name"]))

    return entities


async def update_beacon_on_swipe(
    neo4j_driver: AsyncDriver,
    redis_client,
    user_id: int,
    movie_tmdb_id: int,
    action_type: str,
    is_supercharged: bool,
) -> None:
    """Incrementally update the beacon map in Redis after a swipe."""
    score = SWIPE_SCORES.get((action_type, is_supercharged), 0.0)
    if score == 0.0:
        return

    entities = await _get_movie_entities(neo4j_driver, movie_tmdb_id)
    if not entities:
        return

    redis_key = f"{BEACON_REDIS_PREFIX}{user_id}"

    for entity_type, tmdb_id, name in entities:
        multiplier = ENTITY_MULTIPLIERS.get(entity_type, 0.5)
        delta = score * multiplier
        field = f"{entity_type}:{tmdb_id}"

        existing = await redis_client.hget(redis_key, field)
        if existing:
            entry = json.loads(existing)
            entry["weight"] += delta
        else:
            entry = asdict(
                BeaconEntry(
                    entity_type=entity_type,
                    tmdb_id=tmdb_id,
                    name=name,
                    weight=delta,
                )
            )

        await redis_client.hset(redis_key, field, json.dumps(entry))

    await redis_client.expire(redis_key, BEACON_TTL_SECONDS)


async def load_beacon_map(redis_client, user_id: int) -> BeaconMap | None:
    """Load the beacon map from Redis. Returns None on cache miss."""
    redis_key = f"{BEACON_REDIS_PREFIX}{user_id}"
    data = await redis_client.hgetall(redis_key)
    if not data:
        return None

    beacon_map: BeaconMap = {}
    for field, value in data.items():
        entry_data = json.loads(value)
        entry = BeaconEntry(**entry_data)
        key = (entry.entity_type, entry.tmdb_id)
        beacon_map[key] = entry

    return beacon_map


async def save_beacon_map(redis_client, user_id: int, beacon_map: BeaconMap) -> None:
    """Save a full beacon map to Redis."""
    redis_key = f"{BEACON_REDIS_PREFIX}{user_id}"
    pipe = redis_client.pipeline()
    await pipe.delete(redis_key)
    for (entity_type, tmdb_id), entry in beacon_map.items():
        field = f"{entity_type}:{tmdb_id}"
        pipe.hset(redis_key, field, json.dumps(asdict(entry)))
    pipe.expire(redis_key, BEACON_TTL_SECONDS)
    await pipe.execute()
