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

    def _notify(self, model_name: str, report: dict, elapsed_seconds: float) -> None:
        """Send a Discord training report. Called at the end of run_pipeline()."""
        from movie_recommender.services.notifiers.discord import DiscordNotifier

        try:
            DiscordNotifier().send_training_report(
                model_name=model_name,
                metrics=report.get("metrics", {}),
                config_meta={
                    **report.get("config", {}),
                    "elapsed_min": f"{elapsed_seconds / 60:.2f}",
                    "users_evaluated": report.get("num_users_evaluated", "?"),
                },
            )
        except Exception as e:
            print(f"Connection to Discord failed: {e}")
