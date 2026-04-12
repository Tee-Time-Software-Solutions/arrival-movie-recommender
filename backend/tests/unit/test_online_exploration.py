from unittest.mock import AsyncMock, MagicMock

import pytest

from movie_recommender.schemas.requests.movies import MovieDetails
from movie_recommender.services.recommender.pipeline.feed_manager.main import FeedManager
from movie_recommender.services.recommender.pipeline.online.exploration import (
    EXPLORATION_STATE_TTL_SECONDS,
    get_genre_impression_counts,
    genre_impression_key,
    record_genre_impressions,
)


class TestExplorationState:
    @pytest.mark.asyncio
    async def test_record_genre_impressions_increments_unique_genres(self):
        redis_client = AsyncMock()

        await record_genre_impressions(
            redis_client,
            user_id=7,
            genres=["Action", "Comedy", "Action"],
        )

        key = genre_impression_key(7)
        redis_client.hincrby.assert_any_call(key, "Action", 1)
        redis_client.hincrby.assert_any_call(key, "Comedy", 1)
        assert redis_client.hincrby.await_count == 2
        redis_client.expire.assert_awaited_once_with(
            key, EXPLORATION_STATE_TTL_SECONDS
        )

    @pytest.mark.asyncio
    async def test_record_genre_impressions_skips_empty_input(self):
        redis_client = AsyncMock()

        await record_genre_impressions(redis_client, user_id=7, genres=[])

        redis_client.hincrby.assert_not_called()
        redis_client.expire.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_genre_impression_counts_decodes_redis_payload(self):
        redis_client = AsyncMock()
        redis_client.hgetall.return_value = {b"Action": b"3", "Comedy": 1}

        counts = await get_genre_impression_counts(redis_client, user_id=7)

        assert counts == {"Action": 3, "Comedy": 1}


class TestFeedManagerGenreTracking:
    @pytest.mark.asyncio
    async def test_served_movie_records_genre_impressions(self, synthetic_artifacts):
        recommender = MagicMock()
        recommender.model_artifacts = synthetic_artifacts
        hydrator = AsyncMock()
        hydrator.get_or_fetch_movie.return_value = MovieDetails(
            movie_db_id=100,
            tmdb_id=1000,
            title="Action Movie",
            poster_url="https://example.com/poster.jpg",
            release_year=2020,
            rating=7.5,
            genres=["Action"],
            is_adult=False,
            synopsis="Test synopsis",
            runtime=120,
        )
        redis_client = AsyncMock()
        redis_client.lpop.return_value = "[100, \"Action Movie\"]"
        redis_client.llen.return_value = 10

        feed_manager = FeedManager(
            recommender=recommender,
            hydrator=hydrator,
            redis_client=redis_client,
        )

        movie = await feed_manager.get_next_movie(user_id=42)

        assert movie.title == "Action Movie"
        redis_client.hincrby.assert_awaited_once_with(
            genre_impression_key(42), "Action", 1
        )
        redis_client.expire.assert_awaited_once_with(
            genre_impression_key(42), EXPLORATION_STATE_TTL_SECONDS
        )
