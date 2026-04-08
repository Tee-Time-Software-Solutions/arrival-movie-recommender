"""
KG Traversal — finds explanation paths from a recommended movie to user beacons.

Uses Cypher queries (not programmatic BFS) for efficiency on indexed properties.
Maximum 3 hops, with edge-type priority ordering.
"""

import logging
from dataclasses import dataclass

from neo4j import AsyncDriver

from movie_recommender.services.knowledge_graph.beacon import BeaconMap

logger = logging.getLogger(__name__)

# Edge priority weights (higher = more explanatory power)
EDGE_PRIORITY = {
    "DIRECTED_BY": 3.0,
    "ACTED_IN": 2.5,
    "WRITTEN_BY": 2.0,
    "HAS_GENRE": 1.5,
    "BELONGS_TO": 1.0,
    "HAS_KEYWORD": 0.8,
    "PRODUCED_BY": 0.5,
}


@dataclass
class GraphPath:
    """A path through the KG from the recommended movie to a beacon."""

    hop_count: int
    edge_type: str  # primary edge type (first hop)
    entity_type: str  # type of the beacon entity hit
    entity_tmdb_id: int
    entity_name: str
    # For 2-hop paths: the intermediate movie that connects them
    via_movie_tmdb_id: int | None = None
    via_movie_title: str | None = None


@dataclass
class ScoredPath:
    path: GraphPath
    score: float


async def find_explanation_paths(
    driver: AsyncDriver,
    movie_tmdb_id: int,
    beacon_map: BeaconMap,
    max_hops: int = 3,
) -> list[GraphPath]:
    """Find all explanation paths from the recommended movie to user beacons."""
    if not beacon_map:
        return []

    paths: list[GraphPath] = []

    async with driver.session() as session:
        # Hop 1: Direct connections to beacon entities
        # Directors
        result = await session.run(
            """
            MATCH (m:Movie {tmdb_id: $tmdb_id})-[:DIRECTED_BY]->(p:Person)
            RETURN p.tmdb_id AS tmdb_id, p.name AS name
            """,
            tmdb_id=movie_tmdb_id,
        )
        async for record in result:
            key = ("Director", record["tmdb_id"])
            if key in beacon_map:
                paths.append(
                    GraphPath(
                        hop_count=1,
                        edge_type="DIRECTED_BY",
                        entity_type="Director",
                        entity_tmdb_id=record["tmdb_id"],
                        entity_name=record["name"],
                    )
                )

        # Actors
        result = await session.run(
            """
            MATCH (p:Person)-[:ACTED_IN]->(m:Movie {tmdb_id: $tmdb_id})
            RETURN p.tmdb_id AS tmdb_id, p.name AS name
            """,
            tmdb_id=movie_tmdb_id,
        )
        async for record in result:
            key = ("Actor", record["tmdb_id"])
            if key in beacon_map:
                paths.append(
                    GraphPath(
                        hop_count=1,
                        edge_type="ACTED_IN",
                        entity_type="Actor",
                        entity_tmdb_id=record["tmdb_id"],
                        entity_name=record["name"],
                    )
                )

        # Genres
        result = await session.run(
            """
            MATCH (m:Movie {tmdb_id: $tmdb_id})-[:HAS_GENRE]->(g:Genre)
            RETURN g.tmdb_id AS tmdb_id, g.name AS name
            """,
            tmdb_id=movie_tmdb_id,
        )
        async for record in result:
            key = ("Genre", record["tmdb_id"])
            if key in beacon_map:
                paths.append(
                    GraphPath(
                        hop_count=1,
                        edge_type="HAS_GENRE",
                        entity_type="Genre",
                        entity_tmdb_id=record["tmdb_id"],
                        entity_name=record["name"],
                    )
                )

        # Early exit if we found high-confidence hop-1 paths
        hop1_director = [p for p in paths if p.edge_type == "DIRECTED_BY"]
        hop1_actor = [p for p in paths if p.edge_type == "ACTED_IN"]
        if hop1_director or hop1_actor:
            return paths

        if max_hops < 2:
            return paths

        # Hop 2: Through shared people to other liked movies
        # "This movie shares a director with a movie you liked"
        result = await session.run(
            """
            MATCH (m:Movie {tmdb_id: $tmdb_id})-[:DIRECTED_BY]->(p:Person)<-[:DIRECTED_BY]-(other:Movie)
            WHERE other.tmdb_id <> $tmdb_id
            RETURN p.tmdb_id AS person_id, p.name AS person_name,
                   other.tmdb_id AS movie_id, other.title AS movie_title
            """,
            tmdb_id=movie_tmdb_id,
        )
        async for record in result:
            # Check if the user liked the "other" movie (it should be in beacon data)
            # We check if this person is a beacon (user interacted with movies by them)
            key = ("Director", record["person_id"])
            if key in beacon_map:
                paths.append(
                    GraphPath(
                        hop_count=2,
                        edge_type="DIRECTED_BY",
                        entity_type="Director",
                        entity_tmdb_id=record["person_id"],
                        entity_name=record["person_name"],
                        via_movie_tmdb_id=record["movie_id"],
                        via_movie_title=record["movie_title"],
                    )
                )

        # Shared actors via other movies
        result = await session.run(
            """
            MATCH (p:Person)-[:ACTED_IN]->(m:Movie {tmdb_id: $tmdb_id})
            MATCH (p)-[:ACTED_IN]->(other:Movie)
            WHERE other.tmdb_id <> $tmdb_id
            RETURN p.tmdb_id AS person_id, p.name AS person_name,
                   other.tmdb_id AS movie_id, other.title AS movie_title
            LIMIT 20
            """,
            tmdb_id=movie_tmdb_id,
        )
        async for record in result:
            key = ("Actor", record["person_id"])
            if key in beacon_map:
                paths.append(
                    GraphPath(
                        hop_count=2,
                        edge_type="ACTED_IN",
                        entity_type="Actor",
                        entity_tmdb_id=record["person_id"],
                        entity_name=record["person_name"],
                        via_movie_tmdb_id=record["movie_id"],
                        via_movie_title=record["movie_title"],
                    )
                )

        if paths:
            return paths

        if max_hops < 3:
            return paths

        # Hop 3: Genre combination fallback
        # "Matches your love of Sci-Fi and Thriller"
        result = await session.run(
            """
            MATCH (m:Movie {tmdb_id: $tmdb_id})-[:HAS_GENRE]->(g:Genre)
            RETURN g.tmdb_id AS tmdb_id, g.name AS name
            """,
            tmdb_id=movie_tmdb_id,
        )
        async for record in result:
            key = ("Genre", record["tmdb_id"])
            if key in beacon_map:
                paths.append(
                    GraphPath(
                        hop_count=3,
                        edge_type="HAS_GENRE",
                        entity_type="Genre",
                        entity_tmdb_id=record["tmdb_id"],
                        entity_name=record["name"],
                    )
                )

    return paths


def score_paths(paths: list[GraphPath], beacon_map: BeaconMap) -> list[ScoredPath]:
    """Rank candidate explanation paths by combined beacon weight and edge priority."""
    scored = []
    for path in paths:
        key = (path.entity_type, path.entity_tmdb_id)
        beacon_weight = beacon_map[key].weight if key in beacon_map else 0.0
        edge_weight = EDGE_PRIORITY.get(path.edge_type, 0.5)
        score = (beacon_weight * edge_weight) / path.hop_count
        scored.append(ScoredPath(path=path, score=score))

    scored.sort(key=lambda sp: sp.score, reverse=True)
    return scored
