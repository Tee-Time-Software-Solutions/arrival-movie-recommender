# NOTE: This pipeline previously used LightFM, but LightFM can fail to build on
# some platforms due to Cython/compiler issues. We now use `implicit` BPR.

import time

import movie_recommender.services.recommender.pipeline.offline.models.base.steps.preprocess_movies as preprocess_movies
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.preprocess_ratings as preprocess_ratings
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.fetch_app_swipes as fetch_app_swipes
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.merge_interactions as merge_interactions
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.filter as filter_step
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.prune_movies as prune_movies
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.split as split_step
import movie_recommender.services.recommender.pipeline.offline.models.bpr.steps.data as bpr_data
import movie_recommender.services.recommender.pipeline.offline.models.bpr.steps.trainer as trainer
import movie_recommender.services.recommender.pipeline.offline.models.bpr.steps.evaluator as evaluator
from movie_recommender.services.recommender.pipeline.offline.models.base.base_pipeline import (
    RecommenderPipeline,
)
from movie_recommender.services.recommender.utils.schema import load_config


class BPRPipeline(RecommenderPipeline):
    """
    Second offline model pipeline using implicit BPR (matrix factorization).

    Math:
        Train embeddings with BPR (Bayesian Personalized Ranking) on implicit
        positive-only user↔item interactions. Serve by scoring dot-products
        between user/item latent vectors.
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

        print("\nStep 8: Building BPR data artifacts...")
        bpr_data.run(config)

        print("\nStep 9: Training implicit BPR...")
        trainer.run(config)

        print("\nStep 10: Evaluating BPR model...")
        report = evaluator.run(config)

        elapsed = time.time() - start
        print(f"\n===== PIPELINE COMPLETE ({elapsed / 60:.2f} min) =====")

        self._notify("BPR", report, elapsed)


if __name__ == "__main__":
    BPRPipeline().run_pipeline()
