import time

import movie_recommender.services.recommender.pipeline.offline.models.base.steps.preprocess_movies as preprocess_movies
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.preprocess_ratings as preprocess_ratings
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.fetch_app_swipes as fetch_app_swipes
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.merge_interactions as merge_interactions
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.filter as filter_step
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.prune_movies as prune_movies
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.split as split_step
import movie_recommender.services.recommender.pipeline.offline.models.als.steps.matrix as matrix
import movie_recommender.services.recommender.pipeline.offline.models.als.steps.train_als as train_als
import movie_recommender.services.recommender.pipeline.offline.models.als.steps.metrics as metrics
from movie_recommender.services.recommender.pipeline.offline.models.base.base_pipeline import (
    RecommenderPipeline,
)
from movie_recommender.services.recommender.utils.schema import load_config


class ALSPipeline(RecommenderPipeline):
    """
    Offline ALS pipeline (preprocess → fetch swipes → merge → … → train).

    Math:
        R ≈ U·Vᵀ  (user × movie preference matrix)
        Confidence: C(u,i) = 1 + α·|pref(u,i)|
        Loss: Σ C(u,i)(p(u,i) − uᵤᵀvᵢ)² + λ(‖U‖²+‖V‖²)
        Solved via alternating closed-form least squares.
    """

    def run_pipeline(self) -> None:
        start = time.time()
        config = load_config()

        print(f"\n===== ALS pipeline START =====\n")

        print("Step 1: Preprocessing movies...")
        preprocess_movies.run(config)

        print("\nStep 2: Preprocessing ratings...")
        preprocess_ratings.run(config)

        print("\nStep 3: Fetching app swipes from Postgres → raw parquet...")
        fetch_app_swipes.run(config)

        print("\nStep 4: Merging MovieLens ratings with app swipes...")
        merge_interactions.run(config)

        print("\nStep 5: Filtering sparse users/movies...")
        filter_step.run(config)

        print("\nStep 6: Pruning movie metadata to interaction set...")
        prune_movies.run(config)

        print("\nStep 7: Chronological split...")
        split_step.run(config)

        print("\nStep 8: Building sparse matrix...")
        matrix.run(config)

        print("\nStep 9: Training ALS...")
        train_als.run(config)

        print("\nStep 10: Evaluating model...")
        report = metrics.run(config)

        elapsed = time.time() - start
        print(f"\n===== PIPELINE COMPLETE ({elapsed / 60:.2f} min) =====")

        self._notify("ALS", report, elapsed)


if __name__ == "__main__":
    ALSPipeline().run_pipeline()


def run_pipeline_cron_job() -> None:
    """APScheduler entry point. Uses a Redis lock to prevent concurrent runs across workers."""
    from movie_recommender.core.clients.redis import RedisClient

    redis_client = RedisClient().get_sync_client()
    lock_acquired = redis_client.set("ml_pipeline_lock", "locked", nx=True, ex=3600)

    if not lock_acquired:
        print("Another worker is already running the pipeline. Skipping.")
        return

    try:
        ALSPipeline().run_pipeline()
    finally:
        redis_client.delete("ml_pipeline_lock")
