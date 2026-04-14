"""Unit tests for ``database.CRUD.movies``."""

from movie_recommender.database.CRUD.movies import (
    _get_or_create_crew_person,
    _get_or_create_genre,
    _get_or_create_provider,
    get_filtered_movie_ids,
    get_movie_by_id,
    get_movie_by_tmdb_id,
    get_movies_by_tmdb_ids,
    get_onboarding_movie_cards,
    movie_to_details,
    movies_to_details_bulk,
)
from movie_recommender.schemas.requests.movies import CastMember
from tests.unit.database.CRUD.conftest import MockResult, row


class TestGetFilteredMovieIds:
    async def test_no_filters(self, db):
        db.execute.return_value = MockResult(rows=[row(id=1), row(id=2)])

        ids = await get_filtered_movie_ids(db)

        assert ids == {1, 2}

    async def test_with_all_filters(self, db):
        db.execute.return_value = MockResult(rows=[row(id=5)])

        ids = await get_filtered_movie_ids(
            db, genre_names=["Sci-Fi"], min_year=1990, max_year=2020
        )

        assert ids == {5}


async def test_get_movie_by_tmdb_id_returns_row(db):
    expected = row(id=1, tmdb_id=329865)
    db.execute.return_value = MockResult(rows=[expected])

    assert await get_movie_by_tmdb_id(db, 329865) is expected


class TestGetMoviesByTmdbIds:
    async def test_empty_input_skips_query(self, db):
        assert await get_movies_by_tmdb_ids(db, []) == []
        db.execute.assert_not_awaited()

    async def test_returns_list_of_rows(self, db):
        rows = [row(id=1, tmdb_id=100), row(id=2, tmdb_id=200)]
        db.execute.return_value = MockResult(rows=rows)

        result = await get_movies_by_tmdb_ids(db, [100, 200])

        assert result == rows


async def test_get_movie_by_id_returns_row(db):
    expected = row(id=42)
    db.execute.return_value = MockResult(rows=[expected])

    assert await get_movie_by_id(db, 42) is expected


class TestGetOnboardingMovieCards:
    async def test_empty_input_skips_query(self, db):
        assert await get_onboarding_movie_cards(db, []) == []
        db.execute.assert_not_awaited()

    async def test_builds_cards_with_genres(self, db):
        movie_rows = [
            row(
                id=1,
                tmdb_id=100,
                title="A",
                poster_url="http://a",
                release_year=2020,
                tmdb_rating=7.5,
            ),
            row(
                id=2,
                tmdb_id=200,
                title="B",
                poster_url="http://b",
                release_year=None,
                tmdb_rating=None,
            ),
        ]
        genre_rows = [
            row(movie_id=1, name="Sci-Fi"),
            row(movie_id=1, name="Drama"),
            row(movie_id=2, name="Horror"),
        ]
        db.execute.side_effect = [
            MockResult(rows=movie_rows),
            MockResult(rows=genre_rows),
        ]

        cards = await get_onboarding_movie_cards(db, [100, 200])

        assert len(cards) == 2
        assert cards[0]["movie_db_id"] == 1
        assert cards[0]["genres"] == ["Sci-Fi", "Drama"]
        # None fallbacks
        assert cards[1]["release_year"] == 0
        assert cards[1]["tmdb_rating"] == 0.0
        assert cards[1]["genres"] == ["Horror"]


async def test_movie_to_details_builds_schema(db):
    movie_row = row(
        id=1,
        tmdb_id=329865,
        title="Arrival",
        poster_url="http://p",
        release_year=2016,
        tmdb_rating=7.9,
        is_adult=False,
        synopsis="Linguist meets aliens.",
        runtime=116,
        trailer_url="http://t",
    )
    db.execute.side_effect = [
        MockResult(rows=[movie_row]),
        MockResult(rows=[row(name="Sci-Fi")]),
        MockResult(
            rows=[
                row(
                    name="Amy Adams",
                    role_type="Actor",
                    character_name="Louise",
                    image_url="http://i",
                )
            ]
        ),
        MockResult(rows=[row(name="Netflix", provider_type="flatrate")]),
    ]

    details = await movie_to_details(db, 1)

    assert details.movie_db_id == 1
    assert details.title == "Arrival"
    assert details.genres == ["Sci-Fi"]
    assert len(details.cast) == 1
    assert details.cast[0].name == "Amy Adams"
    assert len(details.movie_providers) == 1
    assert details.movie_providers[0].name == "Netflix"


class TestMoviesToDetailsBulk:
    async def test_empty_input_skips_queries(self, db):
        assert await movies_to_details_bulk(db, []) == []
        db.execute.assert_not_awaited()

    async def test_builds_schemas_and_applies_fallbacks(self, db):
        movie_rows = [
            row(
                id=1,
                tmdb_id=100,
                title="A",
                poster_url="http://a",
                release_year=2020,
                tmdb_rating=8.0,
                is_adult=False,
                synopsis="s1",
                runtime=120,
                trailer_url=None,
            ),
            row(
                id=2,
                tmdb_id=200,
                title="B",
                poster_url=None,
                release_year=None,
                tmdb_rating=None,
                is_adult=None,
                synopsis=None,
                runtime=None,
                trailer_url=None,
            ),
        ]
        db.execute.side_effect = [
            MockResult(rows=movie_rows),
            MockResult(rows=[row(movie_id=1, name="Sci-Fi")]),
            MockResult(
                rows=[
                    row(
                        movie_id=1,
                        name="Actor1",
                        role_type="Actor",
                        character_name="C",
                        image_url=None,
                    )
                ]
            ),
            MockResult(
                rows=[row(movie_id=1, name="Netflix", provider_type="flatrate")]
            ),
        ]

        results = await movies_to_details_bulk(db, [1, 2, 3])

        # Only 1 and 2 present; 3 missing and silently skipped.
        assert len(results) == 2
        assert results[0].genres == ["Sci-Fi"]
        assert results[0].cast[0].name == "Actor1"
        assert results[0].movie_providers[0].name == "Netflix"
        # Fallbacks on the sparse row.
        assert results[1].poster_url == ""
        assert results[1].release_year == 0
        assert results[1].rating == 0.0
        assert results[1].synopsis == ""
        assert results[1].runtime == 0
        assert results[1].genres == []


class TestGetOrCreateGenre:
    async def test_returns_existing(self, db):
        db.execute.return_value = MockResult(rows=[row(id=11)])

        assert await _get_or_create_genre(db, "Sci-Fi") == 11

    async def test_creates_new(self, db):
        db.execute.side_effect = [
            MockResult(rows=[]),
            MockResult(scalar_value=22),
        ]

        assert await _get_or_create_genre(db, "NewGenre") == 22


class TestGetOrCreateCrewPerson:
    async def test_finds_by_tmdb_person_id(self, db):
        member = CastMember(name="Amy", role_type="Actor", tmdb_person_id=123)
        db.execute.return_value = MockResult(rows=[row(id=5)])

        assert await _get_or_create_crew_person(db, member) == 5

    async def test_inserts_when_no_tmdb_person_id(self, db):
        member = CastMember(name="Unknown", role_type="Actor")
        db.execute.return_value = MockResult(scalar_value=7)

        assert await _get_or_create_crew_person(db, member) == 7

    async def test_inserts_when_tmdb_person_id_not_found(self, db):
        member = CastMember(name="Amy", role_type="Actor", tmdb_person_id=999)
        db.execute.side_effect = [
            MockResult(rows=[]),
            MockResult(scalar_value=8),
        ]

        assert await _get_or_create_crew_person(db, member) == 8


class TestGetOrCreateProvider:
    async def test_returns_existing(self, db):
        db.execute.return_value = MockResult(rows=[row(id=3)])

        assert await _get_or_create_provider(db, "Netflix", "flatrate") == 3

    async def test_creates_new(self, db):
        db.execute.side_effect = [
            MockResult(rows=[]),
            MockResult(scalar_value=9),
        ]

        assert await _get_or_create_provider(db, "Disney+", "flatrate") == 9
