from typing import Optional

from movie_recommender.services.recommender.serving.artifact_loader import (
    RecommenderArtifacts,
)


def require_artifacts(
    artifacts: Optional[RecommenderArtifacts], artifact_load_error: Optional[str]
) -> RecommenderArtifacts:
    if artifacts is None:
        raise RuntimeError(
            "Recommender artifacts are not available. "
            f"Details: {artifact_load_error}"
        )
    return artifacts
