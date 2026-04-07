import pytest
from pydantic import ValidationError

from movie_recommender.schemas.requests.users import (
    UserAnalytics,
    UserCreate,
    UserPreferences,
    UserProfileSummary,
    UserDisplayInfo,
)


class TestUserPreferences:
    def test_defaults(self):
        prefs = UserPreferences()
        assert prefs.included_genres == []
        assert prefs.excluded_genres == []
        assert prefs.min_release_year is None
        assert prefs.max_release_year is None
        assert prefs.min_rating is None
        assert prefs.include_adult is False
        assert prefs.movie_providers == []

    def test_full_preferences(self):
        prefs = UserPreferences(
            included_genres=["Sci-Fi", "Drama"],
            excluded_genres=["Horror"],
            min_release_year=2000,
            max_release_year=2025,
            min_rating=7.0,
            include_adult=True,
        )
        assert prefs.included_genres == ["Sci-Fi", "Drama"]
        assert prefs.max_release_year == 2025
        assert prefs.include_adult is True


class TestUserAnalytics:
    def test_valid_analytics(self):
        stats = UserAnalytics(
            total_swipes=100,
            total_likes=60,
            total_dislikes=40,
            total_seen=80,
            top_genres=["Sci-Fi", "Drama"],
        )
        assert stats.total_swipes == 100

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            UserAnalytics(total_swipes=10, total_likes=5, total_dislikes=5)


class TestUserCreate:
    def test_valid_creation(self):
        user = UserCreate(
            firebase_uid="abc123",
            profile_image_url="https://example.com/pic.jpg",
            email="test@example.com",
        )
        assert user.firebase_uid == "abc123"

    def test_missing_email_raises(self):
        with pytest.raises(ValidationError):
            UserCreate(
                firebase_uid="abc123", profile_image_url="https://example.com/pic.jpg"
            )


class TestUserProfileSummary:
    def test_assembles_correctly(self):
        summary = UserProfileSummary(
            profile=UserDisplayInfo(
                username="test",
                avatar_url="https://example.com/avatar.png",
                joined_at="2024-01-01",
            ),
            stats=UserAnalytics(
                total_swipes=10,
                total_likes=6,
                total_dislikes=4,
                total_seen=8,
                top_genres=["Drama"],
            ),
            preferences=UserPreferences(),
        )
        assert summary.profile.username == "test"
        assert summary.stats.total_likes == 6
