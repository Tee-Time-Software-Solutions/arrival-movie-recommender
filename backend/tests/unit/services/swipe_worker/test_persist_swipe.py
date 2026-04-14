"""Unit tests for ``services.swipe_worker.main.persist_swipe``."""

from unittest.mock import AsyncMock, MagicMock, patch

from movie_recommender.services.swipe_worker.main import persist_swipe


def _make_session_factory():
    """Build an async-context-manager factory yielding a mock session."""
    session = MagicMock()
    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=None)
    factory = MagicMock(return_value=session_ctx)
    return factory, session


@patch(
    "movie_recommender.services.swipe_worker.main.create_swipe", new_callable=AsyncMock
)
@patch("movie_recommender.services.swipe_worker.main.DatabaseEngine")
async def test_persist_swipe_writes_event(mock_engine_cls, mock_create_swipe):
    factory, session = _make_session_factory()
    mock_engine_cls.return_value.session_factory = factory

    await persist_swipe(
        user_id=1, movie_id=42, action_type="like", is_supercharged=False
    )

    mock_create_swipe.assert_awaited_once_with(
        db=session,
        user_id=1,
        movie_id=42,
        action_type="like",
        is_supercharged=False,
    )


@patch(
    "movie_recommender.services.swipe_worker.main.create_swipe", new_callable=AsyncMock
)
@patch("movie_recommender.services.swipe_worker.main.DatabaseEngine")
async def test_persist_swipe_swallows_exceptions(mock_engine_cls, mock_create_swipe):
    factory, _ = _make_session_factory()
    mock_engine_cls.return_value.session_factory = factory
    mock_create_swipe.side_effect = RuntimeError("db down")

    # Must not raise.
    await persist_swipe(
        user_id=2, movie_id=7, action_type="dislike", is_supercharged=True
    )

    mock_create_swipe.assert_awaited_once()
