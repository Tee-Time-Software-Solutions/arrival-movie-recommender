"""
Explanation Renderer — converts a scored graph path into a human-readable sentence.

Entity references use @EntityName markers that the frontend renders as bold blue
clickable spans with popup details.
"""

import logging
from dataclasses import dataclass

from movie_recommender.services.knowledge_graph.traversal import ScoredPath

logger = logging.getLogger(__name__)


@dataclass
class EntityReference:
    entity_type: str  # "Person", "Genre", "Movie", etc.
    tmdb_id: int
    name: str


@dataclass
class ExplanationResult:
    text: str
    entities: list[EntityReference]
    confidence: float


def render_explanation(scored_path: ScoredPath) -> ExplanationResult:
    """Convert a scored graph path into a human-readable explanation."""
    path = scored_path.path
    entities: list[EntityReference] = []

    entity_ref = EntityReference(
        entity_type=path.entity_type,
        tmdb_id=path.entity_tmdb_id,
        name=path.entity_name,
    )
    entities.append(entity_ref)

    # Confidence: normalize score to 0-1 range (capped)
    confidence = min(abs(scored_path.score) / 10.0, 1.0)

    if path.hop_count == 1:
        if path.edge_type == "DIRECTED_BY":
            text = f"Directed by @{path.entity_name}, who also directed movies you loved"
        elif path.edge_type == "ACTED_IN":
            text = f"Stars @{path.entity_name}, who you've enjoyed in other films"
        elif path.edge_type == "WRITTEN_BY":
            text = f"Written by @{path.entity_name}, whose work you've enjoyed"
        elif path.edge_type == "HAS_GENRE":
            text = f"Matches your love of @{path.entity_name}"
        else:
            text = f"Connected to @{path.entity_name}"

    elif path.hop_count == 2:
        if path.via_movie_title and path.via_movie_tmdb_id:
            via_ref = EntityReference(
                entity_type="Movie",
                tmdb_id=path.via_movie_tmdb_id,
                name=path.via_movie_title,
            )
            entities.append(via_ref)

            if path.edge_type == "DIRECTED_BY":
                text = (
                    f"Directed by @{path.entity_name}, "
                    f"who also directed @{path.via_movie_title} which you loved"
                )
            elif path.edge_type == "ACTED_IN":
                text = (
                    f"Features @{path.entity_name} "
                    f"from @{path.via_movie_title} which you liked"
                )
            else:
                text = (
                    f"Connected to @{path.entity_name} "
                    f"via @{path.via_movie_title}"
                )
        else:
            if path.edge_type == "DIRECTED_BY":
                text = f"Directed by @{path.entity_name}, who also directed movies you loved"
            elif path.edge_type == "ACTED_IN":
                text = f"Stars @{path.entity_name}, who you've enjoyed in other films"
            else:
                text = f"Connected to @{path.entity_name}"

    else:
        # Hop 3: genre fallback
        text = f"Matches your love of @{path.entity_name}"

    return ExplanationResult(text=text, entities=entities, confidence=confidence)
