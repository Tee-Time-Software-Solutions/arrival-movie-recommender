"""
Unit tests for TMDBFetcher._extract_cast_and_crew (used by MovieHydrator).

Only tests the pure extraction logic — no TMDB HTTP calls or DB access.
"""

from unittest.mock import patch

from movie_recommender.services.recommender.pipeline.hydrator.main import TMDBFetcher


def _make_tmdb_fetcher() -> TMDBFetcher:
    """Create a TMDBFetcher with mocked settings (no real AppSettings or HTTP)."""
    with patch.object(TMDBFetcher, "__init__", lambda self: None):
        fetcher = TMDBFetcher.__new__(TMDBFetcher)
        fetcher.IMG_URL = "https://image.tmdb.org/t/p/w500"
        return fetcher


SAMPLE_CREDITS = {
    "credits": {
        "cast": [
            {
                "name": "Amy Adams",
                "character": "Louise Banks",
                "profile_path": "/amy.jpg",
            },
            {
                "name": "Jeremy Renner",
                "character": "Ian Donnelly",
                "profile_path": "/jeremy.jpg",
            },
            {
                "name": "Forest Whitaker",
                "character": "Colonel Weber",
                "profile_path": None,
            },
            {
                "name": "Michael Stuhlbarg",
                "character": "Agent Halpern",
                "profile_path": "/michael.jpg",
            },
            {
                "name": "Mark O'Brien",
                "character": "Captain Marks",
                "profile_path": "/mark.jpg",
            },
            {
                "name": "Extra Person",
                "character": "Extra",
                "profile_path": "/extra.jpg",
            },
        ],
        "crew": [
            {
                "name": "Denis Villeneuve",
                "job": "Director",
                "profile_path": "/denis.jpg",
            },
            {"name": "Shawn Levy", "job": "Producer", "profile_path": "/shawn.jpg"},
            {"name": "Dan Levine", "job": "Producer", "profile_path": "/dan.jpg"},
            {"name": "Aaron Ryder", "job": "Producer", "profile_path": "/aaron.jpg"},
            {
                "name": "Fourth Producer",
                "job": "Producer",
                "profile_path": "/fourth.jpg",
            },
            {
                "name": "Johann Johannsson",
                "job": "Original Music Composer",
                "profile_path": "/johann.jpg",
            },
        ],
    }
}


class TestExtractCastAndCrew:
    def test_takes_top_5_actors(self):
        tmdb = _make_tmdb_fetcher()
        members = tmdb._extract_cast_and_crew(SAMPLE_CREDITS)

        actors = [m for m in members if m.role_type == "Actor"]
        assert len(actors) == 5
        assert actors[0].name == "Amy Adams"
        assert actors[0].character_name == "Louise Banks"

    def test_sixth_actor_excluded(self):
        tmdb = _make_tmdb_fetcher()
        members = tmdb._extract_cast_and_crew(SAMPLE_CREDITS)

        names = [m.name for m in members if m.role_type == "Actor"]
        assert "Extra Person" not in names

    def test_catches_all_directors(self):
        tmdb = _make_tmdb_fetcher()
        members = tmdb._extract_cast_and_crew(SAMPLE_CREDITS)

        directors = [m for m in members if m.role_type == "Director"]
        assert len(directors) == 1
        assert directors[0].name == "Denis Villeneuve"
        assert directors[0].character_name is None

    def test_limits_to_3_producers(self):
        tmdb = _make_tmdb_fetcher()
        members = tmdb._extract_cast_and_crew(SAMPLE_CREDITS)

        producers = [m for m in members if m.role_type == "Producer"]
        assert len(producers) == 3
        assert "Fourth Producer" not in [p.name for p in producers]

    def test_directors_and_producers_have_no_character_name(self):
        tmdb = _make_tmdb_fetcher()
        members = tmdb._extract_cast_and_crew(SAMPLE_CREDITS)

        non_actors = [m for m in members if m.role_type != "Actor"]
        assert all(m.character_name is None for m in non_actors)

    def test_profile_path_built_when_present(self):
        tmdb = _make_tmdb_fetcher()
        members = tmdb._extract_cast_and_crew(SAMPLE_CREDITS)

        amy = next(m for m in members if m.name == "Amy Adams")
        assert amy.profile_path == "https://image.tmdb.org/t/p/w500/amy.jpg"

    def test_profile_path_none_when_missing(self):
        tmdb = _make_tmdb_fetcher()
        members = tmdb._extract_cast_and_crew(SAMPLE_CREDITS)

        forest = next(m for m in members if m.name == "Forest Whitaker")
        assert forest.profile_path is None

    def test_empty_credits(self):
        tmdb = _make_tmdb_fetcher()
        members = tmdb._extract_cast_and_crew({"credits": {"cast": [], "crew": []}})
        assert members == []

    def test_missing_credits_key(self):
        tmdb = _make_tmdb_fetcher()
        members = tmdb._extract_cast_and_crew({})
        assert members == []

    def test_ignores_non_director_non_producer_crew(self):
        tmdb = _make_tmdb_fetcher()
        members = tmdb._extract_cast_and_crew(SAMPLE_CREDITS)

        names = [m.name for m in members]
        assert "Johann Johannsson" not in names
