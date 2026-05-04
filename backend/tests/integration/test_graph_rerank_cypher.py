"""Integration test: graph_rerank Cypher executes against a real Neo4j.

Skips cleanly when Neo4j isn't reachable (matches the existing skip-on-no-infra
pattern in conftest_integration.py used for Postgres/Redis fixtures).

Seeds a tiny graph (3 movies, 1 director, 1 actor, 1 genre) inside a unique
namespace per run, runs ``compute_graph_scores`` with a hand-built beacon, and
asserts the per-movie scores match the expected sums.
"""

from __future__ import annotations

import os
import uuid
from typing import AsyncIterator

import pytest

pytest_plugins = ["tests.integration.conftest_integration"]


pytestmark = pytest.mark.integration


def _neo4j_uri() -> str:
    return os.environ.get("NEO4J_URI", "bolt://localhost:7687")


def _neo4j_auth() -> tuple[str, str]:
    return (
        os.environ.get("NEO4J_USERNAME", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "password"),
    )


@pytest.fixture
async def real_neo4j_driver() -> AsyncIterator["object"]:
    """AsyncDriver connected to a real Neo4j; skip on unreachable.

    Yields a driver. Tests are responsible for cleaning up any nodes they
    create — `seed_graph` below uses a unique label suffix per test run.
    """
    neo4j_pkg = pytest.importorskip("neo4j")
    driver = neo4j_pkg.AsyncGraphDatabase.driver(_neo4j_uri(), auth=_neo4j_auth())
    try:
        await driver.verify_connectivity()
    except Exception as exc:
        await driver.close()
        pytest.skip(f"integration services unavailable: neo4j ({exc})")

    try:
        yield driver
    finally:
        await driver.close()


@pytest.fixture
async def seeded_graph(real_neo4j_driver):
    """Seed 3 movies + 1 director + 1 actor + 1 genre and yield the tmdb_ids.

    Uses a per-run unique label suffix so the seeded subgraph never collides
    with real data. The cleanup at the end deletes the test subgraph by label.
    """
    suffix = uuid.uuid4().hex[:8]
    test_label = f"GraphRerankTest_{suffix}"

    # tmdb_ids: pick a high-numbered offset to avoid colliding with real movies.
    base = 9_000_000_000 + int(suffix[:4], 16)
    movie_a, movie_b, movie_c = base + 1, base + 2, base + 3
    director_id = base + 100
    actor_id = base + 101
    genre_id = base + 200

    async with real_neo4j_driver.session() as session:
        await session.run(
            f"""
            CREATE (m1:Movie:{test_label} {{tmdb_id: $a, title: 'A'}})
            CREATE (m2:Movie:{test_label} {{tmdb_id: $b, title: 'B'}})
            CREATE (m3:Movie:{test_label} {{tmdb_id: $c, title: 'C'}})
            CREATE (d:Person:{test_label}  {{tmdb_id: $director_id, name: 'D'}})
            CREATE (a:Person:{test_label}  {{tmdb_id: $actor_id,    name: 'A'}})
            CREATE (g:Genre:{test_label}   {{tmdb_id: $genre_id,    name: 'G'}})
            CREATE (m1)-[:DIRECTED_BY]->(d)
            CREATE (m1)-[:HAS_GENRE]->(g)
            CREATE (a)-[:ACTED_IN]->(m2)
            CREATE (m2)-[:HAS_GENRE]->(g)
            // m3 has no beacon-relevant connections
            """,
            a=movie_a,
            b=movie_b,
            c=movie_c,
            director_id=director_id,
            actor_id=actor_id,
            genre_id=genre_id,
        )

    try:
        yield {
            "movie_a": movie_a,
            "movie_b": movie_b,
            "movie_c": movie_c,
            "director_id": director_id,
            "actor_id": actor_id,
            "genre_id": genre_id,
        }
    finally:
        async with real_neo4j_driver.session() as session:
            await session.run(f"MATCH (n:{test_label}) DETACH DELETE n")


async def test_compute_graph_scores_against_real_neo4j(
    real_neo4j_driver, seeded_graph
):
    from movie_recommender.services.knowledge_graph.beacon import BeaconEntry
    from movie_recommender.services.recommender.pipeline.online.serving.graph_rerank import (
        EDGE_PRIORITY_ACTOR,
        EDGE_PRIORITY_DIRECTOR,
        EDGE_PRIORITY_GENRE,
        compute_graph_scores,
    )

    director_w = 5.0
    actor_w = 3.0
    genre_w = 2.0

    beacon_map = {
        ("Director", seeded_graph["director_id"]): BeaconEntry(
            "Director", seeded_graph["director_id"], "D", director_w
        ),
        ("Actor", seeded_graph["actor_id"]): BeaconEntry(
            "Actor", seeded_graph["actor_id"], "A", actor_w
        ),
        ("Genre", seeded_graph["genre_id"]): BeaconEntry(
            "Genre", seeded_graph["genre_id"], "G", genre_w
        ),
    }

    candidates = [
        seeded_graph["movie_a"],
        seeded_graph["movie_b"],
        seeded_graph["movie_c"],
    ]
    scores = await compute_graph_scores(real_neo4j_driver, candidates, beacon_map)

    expected_a = director_w * EDGE_PRIORITY_DIRECTOR + genre_w * EDGE_PRIORITY_GENRE
    expected_b = actor_w * EDGE_PRIORITY_ACTOR + genre_w * EDGE_PRIORITY_GENRE
    expected_c = 0.0

    assert seeded_graph["movie_a"] in scores
    assert seeded_graph["movie_b"] in scores
    assert seeded_graph["movie_c"] in scores
    assert abs(scores[seeded_graph["movie_a"]] - expected_a) < 1e-5
    assert abs(scores[seeded_graph["movie_b"]] - expected_b) < 1e-5
    assert abs(scores[seeded_graph["movie_c"]] - expected_c) < 1e-5
