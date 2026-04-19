"""LangGraph ReAct agent for the movie chatbot."""

from typing import Callable

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from movie_recommender.core.settings.main import AppSettings
from movie_recommender.services.chatbot.tools import (
    create_search_movies_tool,
    create_taste_profile_tool,
)

SYSTEM_PROMPT = (
    "You are Arrival, a friendly and knowledgeable movie recommendation assistant. "
    "You help users discover movies they'll love based on their tastes and preferences.\n\n"
    "Rules:\n"
    "- Always use the search_movies tool when users ask for movie recommendations or searches.\n"
    "- Always use the get_taste_profile tool when users ask about their preferences or viewing patterns.\n"
    "- When presenting movie results, briefly describe why each movie matches what the user asked for.\n"
    "- Be conversational and enthusiastic about movies.\n"
    "- Never make up movie data — only reference movies returned by your tools.\n"
    "- Keep responses concise — 2-3 sentences of commentary per recommendation batch.\n"
    "- If a tool returns no results, suggest the user try broader criteria.\n"
)


def build_agent(db_session_factory: Callable, user_id: int):
    """Build a LangGraph ReAct agent with movie search and taste profile tools."""
    settings = AppSettings().openrouter

    llm = ChatOpenAI(
        model=settings.model_name,
        openai_api_key=settings.api_key,
        openai_api_base=settings.base_url,
        temperature=0.7,
        streaming=True,
    )

    tools = [
        create_search_movies_tool(db_session_factory, user_id),
        create_taste_profile_tool(db_session_factory, user_id),
    ]

    return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
