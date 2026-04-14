import pandas as pd

from movie_recommender.services.recommender.pipeline.offline.models.base.steps.merge_interactions import (
    merge_and_dedupe_interactions,
)


def test_merge_dedupe_keeps_latest_per_user_movie():
    ml = pd.DataFrame(
        {
            "user_id": [1, 1],
            "movie_id": [10, 10],
            "preference": [1, -1],
            "timestamp": [100, 200],
        }
    )
    sw = pd.DataFrame(
        {
            "user_id": [1],
            "movie_id": [10],
            "preference": [2],
            "timestamp": [300],
        }
    )
    train, skips = merge_and_dedupe_interactions(ml, sw)
    assert len(train) == 1
    assert train.iloc[0]["preference"] == 2
    assert train.iloc[0]["timestamp"] == 300


def test_skips_excluded_from_training_frame_but_in_sidecar():
    ml = pd.DataFrame(
        {
            "user_id": [1],
            "movie_id": [10],
            "preference": [1],
            "timestamp": [1],
        }
    )
    sw = pd.DataFrame(
        {
            "user_id": [10_000_002],
            "movie_id": [20],
            "preference": [0],
            "timestamp": [5],
        }
    )
    train, skips = merge_and_dedupe_interactions(ml, sw)
    assert len(train) == 1
    assert set(train["movie_id"]) == {10}
    assert len(skips) == 1
    assert skips.iloc[0]["movie_id"] == 20
    assert skips.iloc[0]["user_id"] == 10_000_002


def test_app_like_and_ml_same_pair_latest_wins():
    ml = pd.DataFrame(
        {
            "user_id": [10_000_001],
            "movie_id": [99],
            "preference": [1],
            "timestamp": [1],
        }
    )
    sw = pd.DataFrame(
        {
            "user_id": [10_000_001],
            "movie_id": [99],
            "preference": [-2],
            "timestamp": [2],
        }
    )
    train, _ = merge_and_dedupe_interactions(ml, sw)
    assert len(train) == 1
    assert train.iloc[0]["preference"] == -2
