import time

from pathlib import Path

from movie_recommender.services.recommender.data_processing.preprocessing.preprocess_movies import (
    preprocess_movies,
)
from movie_recommender.services.recommender.data_processing.preprocessing.preprocess_movies import (
    PROCESSED_PATH as MOVIES_CLEAN_PATH,
)
from movie_recommender.services.recommender.data_processing.preprocessing.preprocess_ratings import (
    preprocess_ratings,
)
from movie_recommender.services.recommender.data_processing.preprocessing.preprocess_ratings import (
    PROCESSED_PATH as INTERACTIONS_CLEAN_PATH,
)
from movie_recommender.services.recommender.data_processing.preprocessing.filtering import (
    run_filtering,
)
from movie_recommender.services.recommender.data_processing.preprocessing.filtering import (
    PROCESSED_OUTPUT as INTERACTIONS_FILTERED_PATH,
)
from movie_recommender.services.recommender.data_processing.preprocessing.prune_movies import (
    prune_movies,
)
from movie_recommender.services.recommender.data_processing.preprocessing.prune_movies import (
    MOVIES_OUTPUT as MOVIES_FILTERED_PATH,
)
from movie_recommender.services.recommender.data_processing.split import run_split
from movie_recommender.services.recommender.data_processing.split import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
)
from movie_recommender.services.recommender.learning.fm.data import (
    build_lightfm_data,
)
from movie_recommender.services.recommender.learning.fm.data import (
    INTERACTIONS_PATH as FM_INTERACTIONS_PATH,
    ITEM_FEATURES_PATH as FM_ITEM_FEATURES_PATH,
    MAPPINGS_PATH as FM_MAPPINGS_PATH,
    ITEM_FEATURE_INDEX_PATH as FM_ITEM_FEATURE_INDEX_PATH,
)
from movie_recommender.services.recommender.learning.fm.trainer import train_fm
from movie_recommender.services.recommender.learning.fm.trainer import (
    MODEL_PATH as FM_MODEL_PATH,
    MODEL_INFO_PATH as FM_MODEL_INFO_PATH,
)
from movie_recommender.services.recommender.learning.fm.evaluation import evaluate_fm


def _all_exist(paths: list[Path]) -> bool:
    return all(p.exists() for p in paths)


def run_pipeline() -> None:
    start_time = time.time()

    print("\n========== FM OFFLINE TRAINING PIPELINE START ==========\n")

    print("Step 1: Preprocessing movies...")
    if MOVIES_CLEAN_PATH.exists():
        print(f"Using existing artifact on disk: {MOVIES_CLEAN_PATH}")
    else:
        preprocess_movies()

    print("\nStep 2: Preprocessing ratings...")
    if INTERACTIONS_CLEAN_PATH.exists():
        print(f"Using existing artifact on disk: {INTERACTIONS_CLEAN_PATH}")
    else:
        preprocess_ratings()

    print("\nStep 3: Filtering sparse users/movies...")
    if INTERACTIONS_FILTERED_PATH.exists():
        print(f"Using existing artifact on disk: {INTERACTIONS_FILTERED_PATH}")
    else:
        run_filtering()

    print("\nStep 4: Pruning movie metadata to interaction set...")
    if MOVIES_FILTERED_PATH.exists():
        print(f"Using existing artifact on disk: {MOVIES_FILTERED_PATH}")
    else:
        prune_movies()

    print("\nStep 5: Chronological split...")
    if _all_exist([TRAIN_PATH, VAL_PATH, TEST_PATH]):
        print(f"Using existing artifacts on disk: {TRAIN_PATH}, {VAL_PATH}, {TEST_PATH}")
    else:
        run_split()

    print("\nStep 6: Building LightFM data (interactions + item features)...")
    if _all_exist(
        [
            FM_INTERACTIONS_PATH,
            FM_ITEM_FEATURES_PATH,
            FM_MAPPINGS_PATH,
            FM_ITEM_FEATURE_INDEX_PATH,
        ]
    ):
        print(
            "Using existing LightFM artifacts on disk: "
            f"{FM_INTERACTIONS_PATH}, {FM_ITEM_FEATURES_PATH}, {FM_MAPPINGS_PATH}"
        )
    else:
        build_lightfm_data()

    print("\nStep 7: Training FM (LightFM) model...")
    if _all_exist([FM_MODEL_PATH, FM_MODEL_INFO_PATH]):
        print(f"Using existing trained model on disk: {FM_MODEL_PATH}")
    else:
        train_fm()

    print("\nStep 8: Evaluating FM model...")
    evaluate_fm()

    total_time = time.time() - start_time

    print("\n========== FM PIPELINE COMPLETE ==========")
    print(f"Total runtime: {total_time / 60:.2f} minutes")


if __name__ == "__main__":
    run_pipeline()

