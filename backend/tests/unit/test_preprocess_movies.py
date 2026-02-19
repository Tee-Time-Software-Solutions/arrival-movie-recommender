import pytest

from movie_recommender.services.recommender.data_processing.preprocessing.preprocess_movies import (
    extract_year,
    clean_title,
    split_genres,
)


class TestExtractYear:
    def test_standard_title(self):
        assert extract_year("Toy Story (1995)") == 1995

    def test_no_year(self):
        assert extract_year("Some Movie Without Year") is None

    def test_multiple_parenthesized_groups_takes_trailing(self):
        # regex anchors to end with $, so only trailing (YYYY) matches
        assert extract_year("Movie (Special) (2001)") == 2001

    def test_year_not_at_end_returns_none(self):
        assert extract_year("(1995) Toy Story") is None


class TestCleanTitle:
    def test_removes_trailing_year(self):
        assert clean_title("Toy Story (1995)") == "Toy Story"

    def test_no_year_unchanged(self):
        assert clean_title("No Year Here") == "No Year Here"

    def test_extra_whitespace(self):
        assert clean_title("  Toy Story  (1995)") == "Toy Story"

    def test_year_not_at_end_unchanged(self):
        assert clean_title("(1995) Toy Story") == "(1995) Toy Story"


class TestSplitGenres:
    def test_pipe_separated(self):
        assert split_genres("Action|Comedy|Drama") == ["Action", "Comedy", "Drama"]

    def test_no_genres_listed(self):
        assert split_genres("(no genres listed)") == []

    def test_single_genre(self):
        assert split_genres("Drama") == ["Drama"]

    def test_many_genres(self):
        genres = "Action|Adventure|Animation|Comedy|Drama|Fantasy"
        result = split_genres(genres)
        assert len(result) == 6
        assert result[0] == "Action"
        assert result[-1] == "Fantasy"
