import time
from pathlib import Path

import movie_recommender.services.recommender.pipeline.offline.models.base.steps.fetch_app_swipes as fetch_app_swipes
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.filter as filter_step
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.merge_interactions as merge_interactions
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.preprocess_movies as preprocess_movies
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.preprocess_ratings as preprocess_ratings
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.prune_movies as prune_movies
import movie_recommender.services.recommender.pipeline.offline.models.base.steps.split as split_step
import movie_recommender.services.recommender.pipeline.offline.models.svm.steps.data as svm_data
import movie_recommender.services.recommender.pipeline.offline.models.svm.steps.evaluator as evaluator
import movie_recommender.services.recommender.pipeline.offline.models.svm.steps.trainer as trainer
from movie_recommender.services.recommender.pipeline.offline.models.base.base_pipeline import (
    RecommenderPipeline,
)
from movie_recommender.services.recommender.utils.schema import load_config


def _source_dataset_paths(config) -> tuple[Path, Path]:
    source_dir = config.data_dirs.source_dir
    return source_dir / "movies.csv", source_dir / "ratings.csv"


def _svm_base_artifacts_available(config) -> bool:
    required_paths = (
        config.data_dirs.processed_dir / "movies_filtered.parquet",
        config.data_dirs.splits_dir / "train.parquet",
        config.data_dirs.splits_dir / "val.parquet",
    )
    return all(path.exists() for path in required_paths)


class SVMPipeline(RecommenderPipeline):
    """
    Offline linear SVM ranking pipeline.

    Uses the shared preprocessing/splitting base steps, then prepares sparse
    user-item features, trains a LinearSVC model, and evaluates ranking metrics.
    """

    def run_pipeline(self) -> None:
        start = time.time()
        config = load_config()
        movies_csv_path, ratings_csv_path = _source_dataset_paths(config)

        print(f"\n===== {self.__class__.__name__} START =====\n")

        run_base_steps = movies_csv_path.exists() and ratings_csv_path.exists()
        if not run_base_steps and _svm_base_artifacts_available(config):
            print("Step 1-7: Source CSVs not found, reusing existing base artifacts...")
            print(f"  missing: {movies_csv_path}")
            print(f"  missing: {ratings_csv_path}")
        elif not run_base_steps:
            raise FileNotFoundError(
                "MovieLens source files were not found and reusable base artifacts are missing. "
                f"Expected files: {movies_csv_path} and {ratings_csv_path}. "
                "Either add the CSV files to data_dirs.source_dir or run ALS/FM once to create "
                "processed/split artifacts before running SVM."
            )
        else:
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

        print("\nStep 8: Preparing SVM ranking data...")
        svm_data.run(config)

        print("\nStep 9: Training linear SVM...")
        trainer.run(config)

        print("\nStep 10: Evaluating SVM...")
        evaluator.run(config)

        elapsed = time.time() - start
        print(f"\n===== PIPELINE COMPLETE ({elapsed / 60:.2f} min) =====")


if __name__ == "__main__":
    SVMPipeline().run_pipeline()
