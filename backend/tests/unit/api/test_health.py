"""Unit tests for the health API endpoints."""

from unittest.mock import AsyncMock, MagicMock

from movie_recommender.dependencies.neo4j import get_neo4j_driver
from movie_recommender.dependencies.redis import get_async_redis
from movie_recommender.main import app
from tests.unit.api.conftest import AUTH_HEADERS


class TestPing:
    def test_ping_returns_pong(self, client):
        resp = client.get("/api/v1/health/ping")

        assert resp.status_code == 200
        assert resp.json() == {"response": "pong"}

    def test_ping_does_not_require_auth(self, client):
        # No Authorization header — health/ping is public
        resp = client.get("/api/v1/health/ping")
        assert resp.status_code == 200


class TestCheckDependencies:
    def _make_neo4j_driver_mock(self, raises: Exception | None = None) -> MagicMock:
        driver = MagicMock()
        session = MagicMock()
        if raises is not None:
            session.run = AsyncMock(side_effect=raises)
        else:
            session.run = AsyncMock(return_value=None)

        class _SessionCtx:
            async def __aenter__(self_inner):
                return session

            async def __aexit__(self_inner, exc_type, exc, tb):
                return False

        driver.session = MagicMock(return_value=_SessionCtx())
        return driver

    def test_all_healthy(self, client):
        redis_client = AsyncMock()
        redis_client.ping.return_value = True
        neo4j_driver = self._make_neo4j_driver_mock()

        app.dependency_overrides[get_async_redis] = lambda: redis_client
        app.dependency_overrides[get_neo4j_driver] = lambda: neo4j_driver

        try:
            resp = client.get("/api/v1/health/dependencies", headers=AUTH_HEADERS)
        finally:
            app.dependency_overrides.pop(get_async_redis, None)
            app.dependency_overrides.pop(get_neo4j_driver, None)

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["checks"]["redis"] == "ok"
        assert body["checks"]["neo4j"] == "ok"

    def test_redis_down_returns_503(self, client):
        redis_client = AsyncMock()
        redis_client.ping.side_effect = RuntimeError("connection refused")
        neo4j_driver = self._make_neo4j_driver_mock()

        app.dependency_overrides[get_async_redis] = lambda: redis_client
        app.dependency_overrides[get_neo4j_driver] = lambda: neo4j_driver

        try:
            resp = client.get("/api/v1/health/dependencies", headers=AUTH_HEADERS)
        finally:
            app.dependency_overrides.pop(get_async_redis, None)
            app.dependency_overrides.pop(get_neo4j_driver, None)

        assert resp.status_code == 503
        detail = resp.json()["detail"]
        assert detail["status"] == "unhealthy"
        assert "failed" in detail["checks"]["redis"]
        assert detail["checks"]["neo4j"] == "ok"

    def test_neo4j_down_returns_503(self, client):
        redis_client = AsyncMock()
        redis_client.ping.return_value = True
        neo4j_driver = self._make_neo4j_driver_mock(raises=RuntimeError("no bolt"))

        app.dependency_overrides[get_async_redis] = lambda: redis_client
        app.dependency_overrides[get_neo4j_driver] = lambda: neo4j_driver

        try:
            resp = client.get("/api/v1/health/dependencies", headers=AUTH_HEADERS)
        finally:
            app.dependency_overrides.pop(get_async_redis, None)
            app.dependency_overrides.pop(get_neo4j_driver, None)

        assert resp.status_code == 503
        detail = resp.json()["detail"]
        assert detail["status"] == "unhealthy"
        assert detail["checks"]["redis"] == "ok"
        assert "failed" in detail["checks"]["neo4j"]
