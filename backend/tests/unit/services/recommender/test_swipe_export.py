from movie_recommender.services.recommender.pipeline.offline.models.base.steps.fetch_app_swipes import (
    ml_user_id_for_app_user,
    swipe_row_to_preference,
    swipes_to_dataframe,
)


def test_swipe_row_to_preference_matches_feedback_mapping():
    assert swipe_row_to_preference("like", False) == 1
    assert swipe_row_to_preference("like", True) == 2
    assert swipe_row_to_preference("dislike", False) == -1
    assert swipe_row_to_preference("dislike", True) == -2
    assert swipe_row_to_preference("skip", False) == 0


def test_ml_user_id_offset_default():
    off = 10_000_000
    assert ml_user_id_for_app_user(42, off) == 10_000_042


def test_ml_user_id_respects_env(monkeypatch):
    monkeypatch.setenv("APP_USER_ID_OFFSET", "5000000")
    # Import after env set — get_app_user_id_offset reads env at call time
    from movie_recommender.services.recommender.pipeline.offline.models.base.steps import fetch_app_swipes as se

    assert se.get_app_user_id_offset() == 5_000_000
    assert se.ml_user_id_for_app_user(3) == 5_000_003


def test_swipes_to_dataframe_mapping():
    rows = [
        {
            "user_id": 1,
            "movie_id": 100,
            "action_type": "like",
            "is_supercharged": True,
            "created_at": None,
        }
    ]
    df = swipes_to_dataframe(rows, app_user_id_offset=1_000_000)
    assert len(df) == 1
    assert df.iloc[0]["app_user_id"] == 1
    assert df.iloc[0]["user_id"] == 1_000_001
    assert df.iloc[0]["movie_id"] == 100
    assert df.iloc[0]["preference"] == 2
    assert df.iloc[0]["timestamp"] == 0
