"""LangGraph tool definitions for the chatbot agent.

All tools are READ-ONLY — they query the database but never modify it.
"""

import json
from typing import Callable

from langchain_core.tools import tool

from movie_recommender.database.CRUD.chatbot import (
    get_user_taste_profile,
    search_movies_by_criteria,
)
from movie_recommender.database.CRUD.movies import movies_to_details_bulk


def create_search_movies_tool(db_session_factory: Callable, user_id: int):
    """Factory that returns a search_movies tool bound to a DB session factory."""

    @tool
    async def search_movies(
        genre_names: list[str] | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        keyword: str | None = None,
        min_rating: float | None = None,
        limit: int = 8,
    ) -> str:
        """Search for movies in the database by genre, year range, keyword in title/synopsis, and minimum rating.

        Use this tool when the user asks for movie recommendations, searches for specific movies,
        or wants to find movies matching certain criteria. Returns full movie details including
        title, year, rating, synopsis, genres, cast, poster URL, and streaming providers.
        """
        async with db_session_factory() as db:
            movie_ids = await search_movies_by_criteria(
                db,
                genre_names=genre_names,
                min_year=min_year,
                max_year=max_year,
                keyword=keyword,
                min_rating=min_rating,
                limit=limit,
            )
            if not movie_ids:
                return "No movies found matching those criteria."
            details = await movies_to_details_bulk(db, movie_ids)
            return json.dumps(
                [d.model_dump(mode="json") for d in details], default=str
            )

    return search_movies


def create_taste_profile_tool(db_session_factory: Callable, user_id: int):
    """Factory that returns a get_taste_profile tool bound to a DB session and user."""

    @tool
    async def get_taste_profile() -> str:
        """Get the current user's movie taste profile based on their liked and disliked movies.

        Use this tool when the user asks about their preferences, taste patterns,
        what kind of movies they enjoy, or their viewing history summary.
        Returns genre breakdown, top liked movies, year range, and average rating.
        """
        async with db_session_factory() as db:
            profile = await get_user_taste_profile(db, user_id)
            return json.dumps(profile, default=str)

    return get_taste_profile
