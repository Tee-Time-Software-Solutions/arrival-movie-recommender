"""
Explainer Orchestrator — single entry point for generating explanations.

Loads beacon map → traverses KG → scores paths → renders explanation.
"""

import logging

from neo4j import AsyncDriver

from movie_recommender.services.knowledge_graph.beacon import (
    BeaconMap,
    build_beacon_map,
    load_beacon_map,
    save_beacon_map,
)
from movie_recommender.services.knowledge_graph.renderer import (
    ExplanationResult,
    render_explanation,
)
from movie_recommender.services.knowledge_graph.traversal import (
    find_explanation_paths,
    score_paths,
)

logger = logging.getLogger(__name__)

# Minimum confidence threshold to show an explanation
MIN_CONFIDENCE = 0.05


async def explain_recommendation(
    neo4j_driver: AsyncDriver,
    redis_client,
    db_session_factory,
    user_id: int,
    movie_tmdb_id: int,
) -> ExplanationResult | None:
    """
    Generate an explanation for why a movie was recommended to a user.

    Returns None if:
    - User has no swipe history (empty beacon map)
    - No meaningful path found in the KG
    - Explanation confidence is below threshold
    """
    # 1. Load beacon map (Redis cache or full rebuild)
    beacon_map = await load_beacon_map(redis_client, user_id)

    if beacon_map is None:
        async with db_session_factory() as db:
            beacon_map = await build_beacon_map(neo4j_driver, db, user_id)
        if beacon_map:
            await save_beacon_map(redis_client, user_id, beacon_map)

    if not beacon_map:
        return None

    # 2. Traverse the KG to find explanation paths
    paths = await find_explanation_paths(neo4j_driver, movie_tmdb_id, beacon_map)
    if not paths:
        return None

    # 3. Score and rank paths
    scored_paths = score_paths(paths, beacon_map)
    if not scored_paths:
        return None

    # 4. Render the best explanation
    best = scored_paths[0]
    explanation = render_explanation(best)

    if explanation.confidence < MIN_CONFIDENCE:
        return None

    return explanation
