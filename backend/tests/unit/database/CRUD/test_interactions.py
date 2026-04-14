"""Unit tests for ``database.CRUD.interactions``."""

from movie_recommender.database.CRUD.interactions import (
    create_swipe,
    get_all_rated_movies,
    get_user_liked_movies,
)
from tests.unit.database.CRUD.conftest import MockResult, row


async def test_create_swipe_returns_inserted_row(db):
    inserted = row(id=1, user_id=1, movie_id=42, action_type="like")
    db.execute.return_value = MockResult(rows=[inserted])

    result = await create_swipe(
        db, user_id=1, movie_id=42, action_type="like", is_supercharged=True
    )

    assert result is inserted
    db.commit.assert_awaited_once()


class TestGetUserLikedMovies:
    async def test_returns_ordered_ids_and_total(self, db):
        db.execute.side_effect = [
            MockResult(scalar_value=2),
            MockResult(rows=[row(movie_id=10), row(movie_id=20)]),
        ]

        ids, total = await get_user_liked_movies(db, user_id=1, limit=5, offset=0)

        assert ids == [10, 20]
        assert total == 2

    async def test_empty_when_no_likes(self, db):
        db.execute.side_effect = [
            MockResult(scalar_value=0),
            MockResult(rows=[]),
        ]

        ids, total = await get_user_liked_movies(db, user_id=1)

        assert ids == []
        assert total == 0


class TestGetAllRatedMovies:
    async def test_returns_all_rated_ids(self, db):
        db.execute.side_effect = [
            MockResult(scalar_value=3),
            MockResult(rows=[row(movie_id=1), row(movie_id=2)]),
        ]

        ids, total = await get_all_rated_movies(db, user_id=1, limit=10, offset=0)

        assert ids == [1, 2]
        assert total == 3

    async def test_empty_when_no_rated(self, db):
        db.execute.side_effect = [
            MockResult(scalar_value=0),
            MockResult(rows=[]),
        ]

        ids, total = await get_all_rated_movies(db, user_id=1)

        assert ids == []
        assert total == 0
