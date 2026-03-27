import pytest
from pydantic import ValidationError

from movie_recommender.schemas.requests.movies import (
    CastMember,
    MovieCard,
    MovieDetails,
    MovieProvider,
    ProviderType,
)


class TestMovieCard:
    def test_valid_card(self):
        card = MovieCard(
            movie_db_id=1,
            tmdb_id=100,
            title="Arrival",
            poster_url="https://example.com/poster.jpg",
            release_year=2016,
            rating=7.9,
            genres=["Drama", "Sci-Fi"],
            is_adult=False,
        )
        assert card.movie_db_id == 1
        assert card.genres == ["Drama", "Sci-Fi"]

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            MovieCard(
                tmdb_id=100,
                title="Arrival",
                poster_url="https://example.com/poster.jpg",
                release_year=2016,
                rating=7.9,
                genres=["Drama"],
                is_adult=False,
            )


class TestCastMember:
    def test_actor_with_character(self):
        member = CastMember(
            name="Amy Adams",
            role_type="Actor",
            character_name="Louise Banks",
            profile_path="https://example.com/profile.jpg",
        )
        assert member.character_name == "Louise Banks"

    def test_director_without_character(self):
        member = CastMember(
            name="Denis Villeneuve",
            role_type="Director",
        )
        assert member.character_name is None
        assert member.profile_path is None

    def test_role_type_is_required(self):
        with pytest.raises(ValidationError):
            CastMember(name="Amy Adams")


class TestMovieProvider:
    def test_valid_provider(self):
        provider = MovieProvider(name="Netflix", provider_type=ProviderType.FLATRATE)
        assert provider.provider_type == ProviderType.FLATRATE

    def test_string_coercion_for_enum(self):
        provider = MovieProvider(name="Apple TV", provider_type="rent")
        assert provider.provider_type == ProviderType.RENT

    def test_invalid_provider_type_raises(self):
        with pytest.raises(ValidationError):
            MovieProvider(name="Netflix", provider_type="stream")


class TestMovieDetails:
    def test_inherits_movie_card_fields(self):
        details = MovieDetails(
            movie_db_id=1,
            tmdb_id=100,
            title="Arrival",
            poster_url="https://example.com/poster.jpg",
            release_year=2016,
            rating=7.9,
            genres=["Drama"],
            is_adult=False,
            synopsis="A linguist is recruited to communicate with aliens.",
            runtime=116,
        )
        assert details.synopsis == "A linguist is recruited to communicate with aliens."
        assert details.cast == []
        assert details.movie_providers == []
        assert details.trailer_url is None

    def test_full_details_with_cast_and_providers(self):
        details = MovieDetails(
            movie_db_id=1,
            tmdb_id=100,
            title="Arrival",
            poster_url="https://example.com/poster.jpg",
            release_year=2016,
            rating=7.9,
            genres=["Drama", "Sci-Fi"],
            is_adult=False,
            synopsis="A linguist is recruited to communicate with aliens.",
            runtime=116,
            trailer_url="https://www.youtube.com/watch?v=abc123",
            cast=[
                CastMember(name="Amy Adams", role_type="Actor", character_name="Louise Banks"),
                CastMember(name="Denis Villeneuve", role_type="Director"),
            ],
            movie_providers=[
                MovieProvider(name="Netflix", provider_type="flatrate"),
            ],
        )
        assert len(details.cast) == 2
        assert details.cast[0].role_type == "Actor"
        assert details.cast[1].character_name is None
