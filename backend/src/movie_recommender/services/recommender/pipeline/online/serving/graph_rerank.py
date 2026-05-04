"""
Knowledge-graph reranking — pulls beacon-weighted graph affinity into the
recommender's score for an ALS top-K shortlist.

The flow is:
  1. ALS scores -> shortlist (top-K) via `als_shortlist`.
  2. For the shortlist, compute per-movie graph scores via `compute_graph_scores`
     using the user's beacon map.
  3. Blend ALS + graph (z-score normalized) via `blend_scores`.

`graph_weight=0` makes the blend a pure pass-through, so the whole module is
no-op when the feature is off.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from neo4j import AsyncDriver

from movie_recommender.services.knowledge_graph.beacon import BeaconMap

logger = logging.getLogger(__name__)

# Edge priorities mirror those used for explanation-path scoring at
# services/knowledge_graph/traversal.py:18-26 — kept in sync by hand to avoid
# importing traversal here (would create a needless cross-module dependency).
EDGE_PRIORITY_DIRECTOR = 3.0
EDGE_PRIORITY_ACTOR = 2.5
EDGE_PRIORITY_WRITER = 2.0
EDGE_PRIORITY_GENRE = 1.5
EDGE_PRIORITY_KEYWORD = 0.8

GRAPH_RERANK_TIMEOUT_S = 0.5

# Pre-built Cypher; using parametrized lists/maps lets Neo4j short-circuit on
# the UNIQUE tmdb_id index.
_GRAPH_SCORE_CYPHER = f"""
UNWIND $candidates AS cand_tmdb_id
MATCH (m:Movie {{tmdb_id: cand_tmdb_id}})
OPTIONAL MATCH (m)-[:DIRECTED_BY]->(d:Person)   WHERE d.tmdb_id IN $director_ids
OPTIONAL MATCH (a:Person)-[:ACTED_IN]->(m)      WHERE a.tmdb_id IN $actor_ids
OPTIONAL MATCH (m)-[:WRITTEN_BY]->(w:Person)    WHERE w.tmdb_id IN $writer_ids
OPTIONAL MATCH (m)-[:HAS_GENRE]->(g:Genre)      WHERE g.tmdb_id IN $genre_ids
OPTIONAL MATCH (m)-[:HAS_KEYWORD]->(k:Keyword)  WHERE k.tmdb_id IN $keyword_ids
WITH cand_tmdb_id,
     sum(coalesce($director_w[toString(d.tmdb_id)], 0.0)) * {EDGE_PRIORITY_DIRECTOR} AS s_dir,
     sum(coalesce($actor_w[toString(a.tmdb_id)],    0.0)) * {EDGE_PRIORITY_ACTOR}    AS s_act,
     sum(coalesce($writer_w[toString(w.tmdb_id)],   0.0)) * {EDGE_PRIORITY_WRITER}   AS s_wri,
     sum(coalesce($genre_w[toString(g.tmdb_id)],    0.0)) * {EDGE_PRIORITY_GENRE}    AS s_gen,
     sum(coalesce($keyword_w[toString(k.tmdb_id)],  0.0)) * {EDGE_PRIORITY_KEYWORD}  AS s_kw
RETURN cand_tmdb_id, s_dir + s_act + s_gen + s_wri + s_kw AS graph_score
"""


def als_shortlist(
    scores: np.ndarray,
    candidate_ids: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices_into_scores, ids) for the top_k highest-scoring candidates."""
    if top_k >= len(scores):
        idx = np.arange(len(scores))
        return idx, candidate_ids

    idx = np.argpartition(scores, -top_k)[-top_k:]
    return idx, candidate_ids[idx]


def _split_beacon_map(
    beacon_map: BeaconMap,
) -> tuple[dict[str, list[int]], dict[str, dict[str, float]]]:
    """Group beacon entries by entity_type into id-lists and string-keyed weight maps.

    Cypher map keys must be strings, so we stringify tmdb_ids when building the
    weight maps.
    """
    type_to_ids: dict[str, list[int]] = {
        "Director": [],
        "Actor": [],
        "Writer": [],
        "Genre": [],
        "Keyword": [],
    }
    type_to_weights: dict[str, dict[str, float]] = {
        "Director": {},
        "Actor": {},
        "Writer": {},
        "Genre": {},
        "Keyword": {},
    }
    for (entity_type, tmdb_id), entry in beacon_map.items():
        if entity_type not in type_to_ids:
            continue
        type_to_ids[entity_type].append(int(tmdb_id))
        type_to_weights[entity_type][str(tmdb_id)] = float(entry.weight)
    return type_to_ids, type_to_weights


async def compute_graph_scores(
    driver: AsyncDriver,
    candidate_tmdb_ids: list[int],
    beacon_map: BeaconMap,
) -> dict[int, float]:
    """Per-movie graph score = sum over beacon-connected entities of weight*edge_priority.

    Negative beacon weights (disliked entities) penalize the candidate by design.
    Returns {} on Neo4j failure or timeout — caller falls back to pure ALS.
    """
    if not candidate_tmdb_ids or not beacon_map:
        return {}

    type_to_ids, type_to_weights = _split_beacon_map(beacon_map)

    params = {
        "candidates": candidate_tmdb_ids,
        "director_ids": type_to_ids["Director"],
        "actor_ids": type_to_ids["Actor"],
        "writer_ids": type_to_ids["Writer"],
        "genre_ids": type_to_ids["Genre"],
        "keyword_ids": type_to_ids["Keyword"],
        "director_w": type_to_weights["Director"],
        "actor_w": type_to_weights["Actor"],
        "writer_w": type_to_weights["Writer"],
        "genre_w": type_to_weights["Genre"],
        "keyword_w": type_to_weights["Keyword"],
    }

    try:
        return await asyncio.wait_for(
            _run_graph_score_query(driver, params),
            timeout=GRAPH_RERANK_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Graph rerank timed out after %.2fs (candidates=%d, beacon_size=%d)",
            GRAPH_RERANK_TIMEOUT_S,
            len(candidate_tmdb_ids),
            len(beacon_map),
        )
        return {}
    except Exception:
        logger.warning("Graph rerank failed — falling back to ALS", exc_info=True)
        return {}


async def _run_graph_score_query(driver: AsyncDriver, params: dict) -> dict[int, float]:
    async with driver.session() as session:
        result = await session.run(_GRAPH_SCORE_CYPHER, **params)
        records = await result.data()
    return {int(r["cand_tmdb_id"]): float(r["graph_score"]) for r in records}


def blend_scores(
    als: np.ndarray,
    graph: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Z-score normalize each side and blend: (1-w)*als_n + w*graph_n.

    weight=0 returns als unchanged. Both arrays must be the same length.
    """
    if weight == 0 or graph.size == 0:
        return als
    als_n = (als - als.mean()) / (als.std() + 1e-8)
    graph_n = (graph - graph.mean()) / (graph.std() + 1e-8)
    return (1.0 - weight) * als_n + weight * graph_n
