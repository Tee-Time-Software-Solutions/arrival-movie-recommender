import numpy as np

from movie_recommender.services.recommender.pipeline.online.serving.ranker import (
    rank_movie_ids,
)


def test_ranker_keeps_original_order_when_exploration_weight_zero(synthetic_artifacts):
    user_vector = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)

    baseline = rank_movie_ids(
        n=5,
        model_artifacts=synthetic_artifacts,
        user_vector=user_vector,
        seen_movie_ids=set(),
    )
    reranked = rank_movie_ids(
        n=5,
        model_artifacts=synthetic_artifacts,
        user_vector=user_vector,
        seen_movie_ids=set(),
        genre_impression_counts={"Action": 20, "Comedy": 1},
        exploration_weight=0.0,
    )

    assert reranked == baseline


def test_ranker_gives_tiny_bonus_to_under_explored_genre(synthetic_artifacts):
    user_vector = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

    ranked = rank_movie_ids(
        n=2,
        model_artifacts=synthetic_artifacts,
        user_vector=user_vector,
        seen_movie_ids={103},
        genre_impression_counts={"Action": 20, "Comedy": 0},
        exploration_weight=0.05,
    )

    assert ranked[0] == 101
    assert ranked[1] == 100
