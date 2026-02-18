import time

from movie_recommender.services.recommender.data_processing.preprocessing.preprocess_movies import preprocess_movies
from movie_recommender.services.recommender.data_processing.preprocessing.preprocess_ratings import preprocess_ratings
from movie_recommender.services.recommender.data_processing.preprocessing.filtering import run_filtering
from movie_recommender.services.recommender.data_processing.split import run_split
from movie_recommender.services.recommender.learning.build_matrix import build_sparse_matrix
from movie_recommender.services.recommender.learning.train_als import train
from movie_recommender.services.recommender.learning.evaluate import evaluate


def run_pipeline():
    start_time = time.time()

    print("\n========== OFFLINE TRAINING PIPELINE START ==========\n")

    print("Step 1: Preprocessing movies...")
    preprocess_movies()

    print("\nStep 2: Preprocessing ratings...")
    preprocess_ratings()

    print("\nStep 3: Filtering sparse users/movies...")
    run_filtering()

    print("\nStep 4: Chronological split...")
    run_split()

    print("\nStep 5: Building sparse matrix...")
    build_sparse_matrix()

    print("\nStep 6: Training ALS...")
    train()

    print("\nStep 7: Evaluating model...")
    evaluate()

    total_time = time.time() - start_time

    print("\n========== PIPELINE COMPLETE ==========")
    print(f"Total runtime: {total_time / 60:.2f} minutes")


if __name__ == "__main__":
    run_pipeline()
