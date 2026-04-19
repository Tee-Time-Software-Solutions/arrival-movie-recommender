from __future__ import annotations

import joblib

from movie_recommender.services.recommender.pipeline.offline.models.svm.steps.data import (
    build_features_for_user_candidates,
    load_svm_training_data,
)
from movie_recommender.services.recommender.pipeline.offline.models.svm.steps.trainer import (
    SVM_MODEL_FILENAME,
)
from movie_recommender.services.recommender.utils.schema import Config


def score_user_movie(user_id: int, movie_id: int, config: Config) -> float:
    """Compute SVM decision score for a (user, movie) pair."""
    assets_dir = config.data_dirs.model_assets_dir
    model = joblib.load(assets_dir / SVM_MODEL_FILENAME)
    _, _, mappings = load_svm_training_data(config)

    features, valid_movies = build_features_for_user_candidates(
        user_id=user_id,
        candidate_movie_ids=[movie_id],
        mappings=mappings,
    )
    if not valid_movies:
        return 0.0
    score = model.decision_function(features)[0]
    return float(score)
