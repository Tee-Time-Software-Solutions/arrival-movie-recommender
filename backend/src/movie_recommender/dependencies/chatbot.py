"""Dependency injection for the chatbot agent."""

from functools import lru_cache
from typing import Callable

from movie_recommender.database.engine import DatabaseEngine
from movie_recommender.services.chatbot.agent import build_agent


class ChatbotAgentFactory:
    """Creates per-user LangGraph agents with DB access."""

    def __init__(self, db_session_factory: Callable):
        self._db_session_factory = db_session_factory

    def __call__(self, user_id: int):
        return build_agent(self._db_session_factory, user_id)


@lru_cache(maxsize=1)
def get_chatbot_agent_factory() -> ChatbotAgentFactory:
    return ChatbotAgentFactory(db_session_factory=DatabaseEngine().session_factory)
