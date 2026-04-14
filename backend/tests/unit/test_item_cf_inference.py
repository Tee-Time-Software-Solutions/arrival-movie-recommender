import numpy as np
from scipy.sparse import csr_matrix

from movie_recommender.services.recommender.pipeline.offline.models.item_cf.steps.inference import (
    recommend_top_n_for_user,
    score_user_movie,
)


def test_item_cf_inference_excludes_seen_items():
    train_matrix = csr_matrix(
        np.array(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    similarity = csr_matrix(
        np.array(
            [
                [0.0, 0.9, 0.2],
                [0.9, 0.0, 0.1],
                [0.2, 0.1, 0.0],
            ],
            dtype=np.float32,
        )
    )
    user_id_to_index = {1: 0, 2: 1}
    movie_id_to_index = {10: 0, 20: 1, 30: 2}
    index_to_movie_id = {0: 10, 1: 20, 2: 30}

    recommendations = recommend_top_n_for_user(
        user_id=1,
        n=2,
        similarity=similarity,
        train_matrix=train_matrix,
        user_id_to_index=user_id_to_index,
        movie_id_to_index=movie_id_to_index,
        index_to_movie_id=index_to_movie_id,
        use_positive_only=True,
        normalize_scores=True,
        exclude_seen=True,
    )

    assert 10 not in recommendations
    assert 30 not in recommendations
    assert recommendations[0] == 20


def test_item_cf_inference_handles_unknown_user_item():
    train_matrix = csr_matrix(np.array([[1.0, 0.0]], dtype=np.float32))
    similarity = csr_matrix(np.array([[0.0, 0.8], [0.8, 0.0]], dtype=np.float32))
    user_id_to_index = {1: 0}
    movie_id_to_index = {10: 0, 20: 1}
    index_to_movie_id = {0: 10, 1: 20}

    unknown_user_recommendations = recommend_top_n_for_user(
        user_id=999,
        n=5,
        similarity=similarity,
        train_matrix=train_matrix,
        user_id_to_index=user_id_to_index,
        movie_id_to_index=movie_id_to_index,
        index_to_movie_id=index_to_movie_id,
        use_positive_only=True,
        normalize_scores=True,
        exclude_seen=True,
    )
    assert unknown_user_recommendations == []

    unknown_item_score = score_user_movie(
        user_id=1,
        movie_id=999,
        similarity=similarity,
        train_matrix=train_matrix,
        user_id_to_index=user_id_to_index,
        movie_id_to_index=movie_id_to_index,
        use_positive_only=True,
        normalize_scores=True,
    )
    assert unknown_item_score == 0.0


def test_item_cf_inference_applies_neighbor_weight_power():
    train_matrix = csr_matrix(np.array([[2.0, 1.0, 0.0]], dtype=np.float32))
    similarity = csr_matrix(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.5, 0.8, 0.0],
            ],
            dtype=np.float32,
        )
    )
    user_id_to_index = {1: 0}
    movie_id_to_index = {10: 0, 20: 1, 30: 2}

    base_score = score_user_movie(
        user_id=1,
        movie_id=30,
        similarity=similarity,
        train_matrix=train_matrix,
        user_id_to_index=user_id_to_index,
        movie_id_to_index=movie_id_to_index,
        use_positive_only=True,
        normalize_scores=True,
        neighbor_weight_power=1.0,
    )
    powered_score = score_user_movie(
        user_id=1,
        movie_id=30,
        similarity=similarity,
        train_matrix=train_matrix,
        user_id_to_index=user_id_to_index,
        movie_id_to_index=movie_id_to_index,
        use_positive_only=True,
        normalize_scores=True,
        neighbor_weight_power=2.0,
    )

    assert base_score > powered_score
