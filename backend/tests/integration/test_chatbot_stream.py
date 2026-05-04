"""
Integration tests for the chatbot SSE endpoint.

These tests boot the full FastAPI app via TestClient (real Postgres,
stubbed Firebase, via the shared ``api_client`` fixture) and override
``get_chatbot_agent_factory`` with a fake agent so we can assert the
SSE wire format without depending on OpenRouter/the real LLM.

Tests are marked ``pytest.mark.integration`` so the existing
``ci-backend-integration`` workflow picks them up automatically.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

import pytest

# Reuse the shared helper fixtures. Mirrors test_api_smoke.py.
pytest_plugins = ["tests.integration.conftest_integration"]

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# SSE parsing helper
# ---------------------------------------------------------------------------


def _parse_sse(text: str) -> list[tuple[str, Any]]:
    """Parse an SSE response body into ``[(event_type, json_data), ...]``.

    Tolerates both ``\\n\\n`` and ``\\r\\n\\r\\n`` block separators.
    Skips blocks that have no ``data:`` line or non-JSON data.
    """
    text = text.replace("\r\n", "\n")
    out: list[tuple[str, Any]] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event_type = ""
        data_str = ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_type = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_str = line[len("data:") :].strip()
        if not event_type or not data_str:
            continue
        try:
            out.append((event_type, json.loads(data_str)))
        except json.JSONDecodeError:
            out.append((event_type, data_str))
    return out


# ---------------------------------------------------------------------------
# Fake agent / factory used to drive deterministic SSE output
# ---------------------------------------------------------------------------


class _FakeChunk:
    """Mimics a LangChain AIMessageChunk well enough for the endpoint."""

    def __init__(self, content: str | list):
        self.content = content


class _FakeAgent:
    def __init__(self, events: Iterable[dict] | None = None, raise_exc: Exception | None = None):
        self._events = list(events or [])
        self._raise_exc = raise_exc

    async def astream_events(self, _payload, version: str = "v2"):
        if self._raise_exc is not None:
            raise self._raise_exc
        for ev in self._events:
            yield ev


class _FakeFactory:
    """Drop-in replacement for ChatbotAgentFactory."""

    def __init__(self, agent: _FakeAgent):
        self._agent = agent

    def __call__(self, user_id: int):  # signature matches the real factory
        return self._agent


def _override_factory(api_client, agent: _FakeAgent):
    from movie_recommender.dependencies.chatbot import get_chatbot_agent_factory
    from movie_recommender.main import app

    app.dependency_overrides[get_chatbot_agent_factory] = lambda: _FakeFactory(agent)


def _clear_factory_override():
    from movie_recommender.dependencies.chatbot import get_chatbot_agent_factory
    from movie_recommender.main import app

    app.dependency_overrides.pop(get_chatbot_agent_factory, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_auth_required(api_client):
    """No Authorization header → 401 (verify_user rejects)."""
    # The shared fixture sets a default Bearer header; strip it for this test.
    saved = api_client.headers.pop("Authorization", None)
    try:
        resp = api_client.post(
            "/api/v1/chatbot/stream", json={"message": "hi"}
        )
        assert resp.status_code == 401, resp.text
    finally:
        if saved is not None:
            api_client.headers["Authorization"] = saved


def test_token_streaming_flow(api_client):
    """Three streamed chunks → three event:token frames followed by event:done."""
    events = [
        {"event": "on_chat_model_stream", "data": {"chunk": _FakeChunk("Hello ")}},
        {"event": "on_chat_model_stream", "data": {"chunk": _FakeChunk("there")}},
        {"event": "on_chat_model_stream", "data": {"chunk": _FakeChunk("!")}},
    ]
    _override_factory(api_client, _FakeAgent(events))
    try:
        resp = api_client.post(
            "/api/v1/chatbot/stream", json={"message": "hi"}
        )
        assert resp.status_code == 200, resp.text
        parsed = _parse_sse(resp.text)
        kinds = [k for k, _ in parsed]
        # Expect: token, token, token, done
        assert kinds == ["token", "token", "token", "done"], kinds
        tokens = [d["token"] for k, d in parsed if k == "token"]
        assert tokens == ["Hello ", "there", "!"]
    finally:
        _clear_factory_override()


def test_token_streaming_handles_content_block_lists(api_client):
    """chunk.content can be a list of blocks; the endpoint should concatenate text blocks."""
    events = [
        {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": _FakeChunk(
                    [{"type": "text", "text": "Hi "}, {"type": "text", "text": "user"}]
                )
            },
        },
    ]
    _override_factory(api_client, _FakeAgent(events))
    try:
        resp = api_client.post(
            "/api/v1/chatbot/stream", json={"message": "hi"}
        )
        assert resp.status_code == 200
        parsed = _parse_sse(resp.text)
        token_events = [d for k, d in parsed if k == "token"]
        assert token_events == [{"token": "Hi user"}]
    finally:
        _clear_factory_override()


def test_search_movies_tool_emits_movies_event(api_client):
    """search_movies tool with a non-empty JSON list → event:movies."""
    fake_movies = [
        {"movie_db_id": 1, "title": "A", "tmdb_id": 100},
        {"movie_db_id": 2, "title": "B", "tmdb_id": 200},
    ]
    events = [
        {
            "event": "on_tool_end",
            "name": "search_movies",
            "data": {"output": json.dumps(fake_movies)},
        },
    ]
    _override_factory(api_client, _FakeAgent(events))
    try:
        resp = api_client.post(
            "/api/v1/chatbot/stream", json={"message": "find me sci-fi"}
        )
        assert resp.status_code == 200
        parsed = _parse_sse(resp.text)
        movies_events = [d for k, d in parsed if k == "movies"]
        assert len(movies_events) == 1
        assert movies_events[0]["movies"] == fake_movies
        # done frame still present
        assert ("done", {}) in parsed
    finally:
        _clear_factory_override()


def test_search_movies_empty_result_skips_movies_event(api_client):
    """The literal 'No movies found...' string must NOT produce a movies event."""
    events = [
        {
            "event": "on_tool_end",
            "name": "search_movies",
            "data": {"output": "No movies found matching those criteria."},
        },
    ]
    _override_factory(api_client, _FakeAgent(events))
    try:
        resp = api_client.post(
            "/api/v1/chatbot/stream", json={"message": "obscure request"}
        )
        assert resp.status_code == 200
        parsed = _parse_sse(resp.text)
        kinds = [k for k, _ in parsed]
        assert "movies" not in kinds
        assert "done" in kinds
    finally:
        _clear_factory_override()


def test_taste_profile_tool_emits_taste_profile_event(api_client):
    """get_taste_profile tool → event:taste_profile with the JSON payload."""
    profile = {
        "total_likes": 12,
        "total_dislikes": 3,
        "genre_counts": [{"genre": "Drama", "count": 5}],
        "top_movies": [],
        "year_range": {"min": 1995, "max": 2024},
        "avg_rating": 7.42,
    }
    events = [
        {
            "event": "on_tool_end",
            "name": "get_taste_profile",
            "data": {"output": json.dumps(profile)},
        },
    ]
    _override_factory(api_client, _FakeAgent(events))
    try:
        resp = api_client.post(
            "/api/v1/chatbot/stream", json={"message": "what do i like?"}
        )
        assert resp.status_code == 200
        parsed = _parse_sse(resp.text)
        taste_events = [d for k, d in parsed if k == "taste_profile"]
        assert len(taste_events) == 1
        assert taste_events[0]["profile"] == profile
    finally:
        _clear_factory_override()


def test_agent_exception_emits_error_event(api_client):
    """Any exception raised in astream_events surfaces as event:error."""
    _override_factory(
        api_client, _FakeAgent(raise_exc=RuntimeError("OpenRouter 502"))
    )
    try:
        resp = api_client.post(
            "/api/v1/chatbot/stream", json={"message": "hi"}
        )
        # The HTTP status of the SSE stream itself stays 200 — the failure is
        # surfaced inside the stream as an `error` frame.
        assert resp.status_code == 200, resp.text
        parsed = _parse_sse(resp.text)
        kinds = [k for k, _ in parsed]
        assert "error" in kinds
        # `done` should NOT appear when the stream errors out.
        assert "done" not in kinds
        error_payload = next(d for k, d in parsed if k == "error")
        assert "OpenRouter 502" in error_payload["error"]
    finally:
        _clear_factory_override()


def test_user_not_found_returns_404(api_client, real_db_engine):
    """If the auth user's firebase_uid has no users row, the endpoint returns 404 before opening the stream."""
    # Temporarily point the firebase stub at a UID that isn't seeded in the DB.
    import firebase_admin.auth as fb_auth

    bogus_uid = "uid-that-does-not-exist-in-postgres"

    saved_verify = fb_auth.verify_id_token
    saved_get_user = fb_auth.get_user

    class _FakeUserRecord:
        email_verified = True
        display_name = "Nobody"

    fb_auth.verify_id_token = lambda *_a, **_k: {"uid": bogus_uid, "email": "n@x.io"}
    fb_auth.get_user = lambda *_a, **_k: _FakeUserRecord()

    # Even with a fake agent in place, the 404 comes from the user lookup
    # *before* the agent is invoked.
    _override_factory(api_client, _FakeAgent(events=[]))
    try:
        resp = api_client.post(
            "/api/v1/chatbot/stream", json={"message": "hi"}
        )
        assert resp.status_code == 404, resp.text
        assert "User not found" in resp.text
    finally:
        _clear_factory_override()
        fb_auth.verify_id_token = saved_verify
        fb_auth.get_user = saved_get_user
