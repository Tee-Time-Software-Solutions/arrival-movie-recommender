import time

from movie_recommender.services.recommender.data_processing.preprocessing.preprocess_movies import (
    preprocess_movies,
)
from movie_recommender.services.recommender.data_processing.preprocessing.preprocess_ratings import (
    preprocess_ratings,
)
from movie_recommender.services.recommender.data_processing.preprocessing.filtering import (
    run_filtering,
)
from movie_recommender.services.recommender.data_processing.preprocessing.prune_movies import (
    prune_movies,
)
from movie_recommender.services.recommender.data_processing.split import run_split
from movie_recommender.services.recommender.learning.fm.data import (
    build_lightfm_data,
)
from movie_recommender.services.recommender.learning.fm.trainer import train_fm
from movie_recommender.services.recommender.learning.fm.evaluation import evaluate_fm


def run_pipeline() -> None:
    start_time = time.time()

    print("\n========== FM OFFLINE TRAINING PIPELINE START ==========\n")

    print("Step 1: Preprocessing movies...")
    preprocess_movies()

    print("\nStep 2: Preprocessing ratings...")
    preprocess_ratings()

    print("\nStep 3: Filtering sparse users/movies...")
    run_filtering()

    print("\nStep 4: Pruning movie metadata to interaction set...")
    prune_movies()

    print("\nStep 5: Chronological split...")
    run_split()

    print("\nStep 6: Building LightFM data (interactions + item features)...")
    build_lightfm_data()

    print("\nStep 7: Training FM (LightFM) model...")
    train_fm()

    print("\nStep 8: Evaluating FM model...")
    evaluate_fm()

    total_time = time.time() - start_time

    print("\n========== FM PIPELINE COMPLETE ==========")
    print(f"Total runtime: {total_time / 60:.2f} minutes")


if __name__ == "__main__":
    run_pipeline()

