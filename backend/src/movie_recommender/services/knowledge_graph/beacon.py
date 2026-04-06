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


PEOPLE_ENTITY_TYPES = {"Director", "Actor", "Writer"}


async def get_top_people(
    redis_client,
    neo4j_driver: AsyncDriver,
    db: AsyncSession,
    user_id: int,
    limit: int = 5,
) -> dict[str, list[dict]]:
    """
    Extract top directors, actors, and writers from the beacon map.
    Returns a dict with keys 'directors', 'actors', 'writers',
    each containing a list of dicts with tmdb_id, name, entity_type, weight,
    image_url, and linked_movies (user's liked movies they appear in).
    """
    beacon_map = await load_beacon_map(redis_client, user_id)
    if beacon_map is None:
        beacon_map = await build_beacon_map(neo4j_driver, db, user_id)
        if beacon_map:
            await save_beacon_map(redis_client, user_id, beacon_map)

    # Filter to people with positive weight, group by type
    grouped: dict[str, list[BeaconEntry]] = {
        "Director": [],
        "Actor": [],
        "Writer": [],
    }
    for entry in beacon_map.values():
        if entry.entity_type in PEOPLE_ENTITY_TYPES and entry.weight > 0:
            grouped[entry.entity_type].append(entry)

    # Sort each group by weight desc, take top N
    for entity_type in grouped:
        grouped[entity_type].sort(key=lambda e: e.weight, reverse=True)
        grouped[entity_type] = grouped[entity_type][:limit]

    # Collect all person tmdb_ids for batch queries
    all_tmdb_ids = [entry.tmdb_id for entries in grouped.values() for entry in entries]

    if not all_tmdb_ids:
        return {"directors": [], "actors": [], "writers": []}

    # Batch fetch person images
    image_map = await _batch_fetch_person_images(neo4j_driver, all_tmdb_ids)

    # Get user's liked movie tmdb_ids from swipes
    liked_tmdb_ids = await _get_liked_movie_tmdb_ids(db, user_id)

    # Batch fetch linked movies (which liked movies each person appears in)
    linked_movies_map: dict[int, list[dict]] = {}
    if liked_tmdb_ids:
        linked_movies_map = await _batch_fetch_linked_movies(
            neo4j_driver, db, all_tmdb_ids, liked_tmdb_ids
        )

    # Build response dicts
    result = {}
    type_to_key = {"Director": "directors", "Actor": "actors", "Writer": "writers"}
    for entity_type, response_key in type_to_key.items():
        result[response_key] = [
            {
                "tmdb_id": entry.tmdb_id,
                "name": entry.name,
                "entity_type": entry.entity_type,
                "weight": round(entry.weight, 2),
                "image_url": image_map.get(entry.tmdb_id),
                "linked_movies": linked_movies_map.get(entry.tmdb_id, []),
            }
            for entry in grouped[entity_type]
        ]

    return result


async def _batch_fetch_person_images(
    neo4j_driver: AsyncDriver, tmdb_ids: list[int]
) -> dict[int, str | None]:
    """Batch-fetch image_url for Person nodes from Neo4j."""
    image_map: dict[int, str | None] = {}
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            UNWIND $tmdb_ids AS tid
            MATCH (p:Person {tmdb_id: tid})
            RETURN p.tmdb_id AS tmdb_id, p.image_url AS image_url
            """,
            tmdb_ids=tmdb_ids,
        )
        records = await result.data()
        for record in records:
            image_map[record["tmdb_id"]] = record["image_url"]
    return image_map


async def get_person_linked_movies(
    neo4j_driver: AsyncDriver,
    db: AsyncSession,
    user_id: int,
    person_tmdb_id: int,
) -> list[dict]:
    """
    For a single person, find which of the user's liked movies they appear in.
    Returns [{tmdb_id, title, poster_url}, ...].
    """
    liked_tmdb_ids = await _get_liked_movie_tmdb_ids(db, user_id)
    if not liked_tmdb_ids:
        return []

    result = await _batch_fetch_linked_movies(
        neo4j_driver, db, [person_tmdb_id], liked_tmdb_ids
    )
    return result.get(person_tmdb_id, [])


async def _get_liked_movie_tmdb_ids(db: AsyncSession, user_id: int) -> list[int]:
    """Get tmdb_ids of movies the user liked (positive swipe score)."""
    result = await db.execute(
        select(movies.c.tmdb_id)
        .join(swipes, swipes.c.movie_id == movies.c.id)
        .where(
            swipes.c.user_id == user_id,
            swipes.c.action_type == "like",
            movies.c.tmdb_id.isnot(None),
        )
        .distinct()
    )
    return [row.tmdb_id for row in result.fetchall()]


async def _batch_fetch_linked_movies(
    neo4j_driver: AsyncDriver,
    db: AsyncSession,
    person_tmdb_ids: list[int],
    liked_tmdb_ids: list[int],
) -> dict[int, list[dict]]:
    """
    For each person, find which of the user's liked movies they appear in.
    Returns {person_tmdb_id: [{tmdb_id, title, poster_url}, ...]}.
    """
    # Query Neo4j for person-movie connections within liked movies
    person_movies: dict[int, list[dict]] = {}
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            UNWIND $person_tmdb_ids AS pid
            MATCH (p:Person {tmdb_id: pid})
            MATCH (p)-[:ACTED_IN|DIRECTED_BY|WRITTEN_BY]-(m:Movie)
            WHERE m.tmdb_id IN $liked_tmdb_ids
            RETURN pid AS person_tmdb_id,
                   collect(DISTINCT {tmdb_id: m.tmdb_id, title: m.title}) AS movies
            """,
            person_tmdb_ids=person_tmdb_ids,
            liked_tmdb_ids=liked_tmdb_ids,
        )
        records = await result.data()
        for record in records:
            person_movies[record["person_tmdb_id"]] = record["movies"]

    # Collect all unique movie tmdb_ids to batch-fetch poster_urls from PostgreSQL
    all_movie_tmdb_ids = set()
    for movie_list in person_movies.values():
        for m in movie_list:
            all_movie_tmdb_ids.add(m["tmdb_id"])

    poster_map: dict[int, str | None] = {}
    if all_movie_tmdb_ids:
        result = await db.execute(
            select(movies.c.tmdb_id, movies.c.poster_url).where(
                movies.c.tmdb_id.in_(list(all_movie_tmdb_ids))
            )
        )
        for row in result.fetchall():
            poster_map[row.tmdb_id] = row.poster_url

    # Enrich with poster_urls
    for person_id, movie_list in person_movies.items():
        person_movies[person_id] = [
            {
                "tmdb_id": m["tmdb_id"],
                "title": m["title"],
                "poster_url": poster_map.get(m["tmdb_id"]),
            }
            for m in movie_list
        ]

    return person_movies
