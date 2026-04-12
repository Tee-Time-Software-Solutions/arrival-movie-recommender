# NOTE: LightFM (pip install lightfm) must be installed for this pipeline to run.
# lightfm currently fails to build on some platforms due to Cython/compiler issues.

import time

import movie_recommender.services.recommender.pipeline.offline.models.base.steps.preprocess_movies as preprocess_movies
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.preprocess_ratings as preprocess_ratings
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.fetch_app_swipes as fetch_app_swipes
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.merge_interactions as merge_interactions
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.filter as filter_step
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.prune_movies as prune_movies
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.split as split_step
import movie_recommender.services.recommender.pipeline.offline.models.fm.steps.data as fm_data
import movie_recommender.services.recommender.pipeline.offline.models.fm.steps.trainer as trainer
import movie_recommender.services.recommender.pipeline.offline.models.fm.steps.evaluator as evaluator
from movie_recommender.services.recommender.pipeline.offline.models.base.base_pipeline import (
    RecommenderPipeline,
)
from movie_recommender.services.recommender.utils.schema import load_config


class FMPipeline(RecommenderPipeline):
    """
    Factorization Machine pipeline using LightFM (BPR loss).

    Math:
        ŷ(x) = w₀ + Σ wⱼxⱼ + Σᵢ Σⱼ₍ᵢ₎ ⟨vᵢ, vⱼ⟩ xᵢ xⱼ
        Loss: BPR (Bayesian Personalised Ranking)

    Requires: pip install lightfm
    """

    def run_pipeline(self) -> None:
        start = time.time()
        config = load_config()

        print(f"\n===== {self.__class__.__name__} START =====\n")

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

        print("\nStep 8: Building LightFM data artifacts...")
        fm_data.run(config)

        print("\nStep 9: Training LightFM (BPR)...")
        trainer.run(config)

        print("\nStep 10: Evaluating LightFM...")
        evaluator.run(config)

        elapsed = time.time() - start
        print(f"\n===== PIPELINE COMPLETE ({elapsed / 60:.2f} min) =====")


if __name__ == "__main__":
    FMPipeline().run_pipeline()
