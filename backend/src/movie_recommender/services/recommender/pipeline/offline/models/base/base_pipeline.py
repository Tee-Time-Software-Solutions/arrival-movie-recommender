from __future__ import annotations

from abc import ABC, abstractmethod

from movie_recommender.services.recommender.utils.schema import Config, load_config


class RecommenderPipeline(ABC):
    """
    Abstract base for offline training pipelines.

    Each subclass owns its full run_pipeline() implementation — shared steps
    (preprocessing, filtering, splitting) are imported and called directly
    in the child. This keeps each pipeline self-contained and explicit.
    """

    @abstractmethod
    def run_pipeline(self) -> None:
        """Run the full training pipeline end-to-end."""
