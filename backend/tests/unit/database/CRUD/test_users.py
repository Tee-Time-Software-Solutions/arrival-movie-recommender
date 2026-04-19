"""Unit tests for ``database.CRUD.users``."""

from movie_recommender.database.CRUD.users import (
    create_user,
    get_user_analytics,
    get_user_by_firebase_uid,
    get_user_excluded_genres,
    get_user_included_genres,
    get_user_preferences,
    mark_onboarding_completed,
    update_user_preferences,
)
from movie_recommender.schemas.requests.users import UserCreate
from tests.unit.database.CRUD.conftest import MockResult, row


async def test_create_user_inserts_and_commits(db):
    inserted = row(id=1, firebase_uid="abc", email="a@b.c")
    db.execute.return_value = MockResult(rows=[inserted])
    payload = UserCreate(
        firebase_uid="abc", profile_image_url="http://x/p.jpg", email="a@b.c"
    )

    result = await create_user(db, payload)

    assert result is inserted
    db.commit.assert_awaited_once()


class TestGetUserByFirebaseUid:
    async def test_returns_row(self, db):
        expected = row(id=1, firebase_uid="abc")
        db.execute.return_value = MockResult(rows=[expected])

        assert await get_user_by_firebase_uid(db, "abc") is expected

    async def test_returns_none_when_missing(self, db):
        db.execute.return_value = MockResult(rows=[])

        assert await get_user_by_firebase_uid(db, "nope") is None


async def test_get_user_preferences_returns_row(db):
    expected = row(user_id=1, min_year=1990)
    db.execute.return_value = MockResult(rows=[expected])

    assert await get_user_preferences(db, 1) is expected


async def test_get_user_included_genres_returns_names(db):
    db.execute.return_value = MockResult(rows=[row(name="Sci-Fi"), row(name="Drama")])

    assert await get_user_included_genres(db, 1) == ["Sci-Fi", "Drama"]


async def test_get_user_excluded_genres_returns_names(db):
    db.execute.return_value = MockResult(rows=[row(name="Horror")])

    assert await get_user_excluded_genres(db, 1) == ["Horror"]


async def test_get_user_analytics_computes_stats(db):
    db.execute.side_effect = [
        MockResult(rows=[row(total_swipes=10, total_likes=6, total_dislikes=3)]),
        MockResult(rows=[row(name="Sci-Fi", cnt=4), row(name="Drama", cnt=2)]),
    ]

    analytics = await get_user_analytics(db, user_id=1)

    assert analytics == {
        "total_swipes": 10,
        "total_likes": 6,
        "total_dislikes": 3,
        "total_seen": 9,
        "top_genres": ["Sci-Fi", "Drama"],
    }


class TestUpdateUserPreferences:
    async def test_inserts_when_missing(self, db):
        db.execute.side_effect = [
            MockResult(rows=[]),  # existing check → empty
            MockResult(rows=[]),  # insert preferences
        ]

        await update_user_preferences(
            db, user_id=1, min_year=1990, include_adult=False
        )

        db.commit.assert_awaited_once()
        assert db.execute.await_count == 2

    async def test_updates_when_exists(self, db):
        db.execute.side_effect = [
            MockResult(rows=[row(id=5)]),  # existing
            MockResult(rows=[]),  # update
        ]

        await update_user_preferences(db, user_id=1, max_year=2025)

        db.commit.assert_awaited_once()
        assert db.execute.await_count == 2

    async def test_skips_update_when_no_values_and_row_exists(self, db):
        db.execute.side_effect = [
            MockResult(rows=[row(id=5)]),  # existing, but no values to change
        ]

        await update_user_preferences(db, user_id=1)

        db.commit.assert_awaited_once()
        assert db.execute.await_count == 1

    async def test_syncs_included_and_excluded_genres(self, db):
        db.execute.side_effect = [
            MockResult(rows=[row(id=5)]),  # existing preferences
            MockResult(rows=[]),  # update preferences
            # _sync_genre_list included (["Sci-Fi"] exists)
            MockResult(rows=[]),  # delete included assoc
            MockResult(rows=[row(id=11)]),  # select genre "Sci-Fi"
            MockResult(rows=[]),  # insert assoc
            # _sync_genre_list excluded (["Horror"] does not exist)
            MockResult(rows=[]),  # delete excluded assoc
            MockResult(rows=[]),  # select genre "Horror" (miss)
            MockResult(scalar_value=22),  # insert genres returning id
            MockResult(rows=[]),  # insert assoc
        ]

        await update_user_preferences(
            db,
            user_id=1,
            min_rating=7.0,
            included_genre_names=["Sci-Fi"],
            excluded_genre_names=["Horror"],
        )

        db.commit.assert_awaited_once()


async def test_mark_onboarding_completed(db):
    db.execute.return_value = MockResult()

    await mark_onboarding_completed(db, user_id=1)

    db.commit.assert_awaited_once()
    db.execute.assert_awaited_once()
