"""Unit tests for ``database.CRUD.watchlist``."""

from sqlalchemy.exc import IntegrityError

from movie_recommender.database.CRUD.watchlist import (
    add_to_watchlist,
    get_user_watchlist,
    remove_from_watchlist,
)
from tests.unit.database.CRUD.conftest import MockResult, row


class TestAddToWatchlist:
    async def test_success_returns_inserted_row(self, db):
        inserted = row(user_id=1, movie_id=42)
        db.execute.return_value = MockResult(rows=[inserted])

        result = await add_to_watchlist(db, user_id=1, movie_id=42)

        assert result is inserted
        db.commit.assert_awaited_once()

    async def test_integrity_error_rolls_back_and_returns_none(self, db):
        db.execute.side_effect = IntegrityError("stmt", {}, Exception("dup"))

        result = await add_to_watchlist(db, user_id=1, movie_id=42)

        assert result is None
        db.rollback.assert_awaited_once()
        db.commit.assert_not_awaited()


class TestRemoveFromWatchlist:
    async def test_removes_existing_row(self, db):
        db.execute.return_value = MockResult(rowcount=1)

        assert await remove_from_watchlist(db, 1, 42) is True
        db.commit.assert_awaited_once()

    async def test_returns_false_when_nothing_removed(self, db):
        db.execute.return_value = MockResult(rowcount=0)

        assert await remove_from_watchlist(db, 1, 42) is False
        db.commit.assert_awaited_once()


class TestGetUserWatchlist:
    async def test_returns_movie_ids_and_count(self, db):
        db.execute.side_effect = [
            MockResult(scalar_value=2),
            MockResult(rows=[row(movie_id=10), row(movie_id=20)]),
        ]

        ids, total = await get_user_watchlist(db, user_id=1, limit=20, offset=0)

        assert ids == [10, 20]
        assert total == 2

    async def test_empty_watchlist(self, db):
        db.execute.side_effect = [
            MockResult(scalar_value=0),
            MockResult(rows=[]),
        ]

        ids, total = await get_user_watchlist(db, user_id=1)

        assert ids == []
        assert total == 0
