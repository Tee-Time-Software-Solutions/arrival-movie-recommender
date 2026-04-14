"""
Opt-in integration test helper fixtures.

This module is a *helper* that tests explicitly import (or merge via
``pytest_plugins``) — it is deliberately NOT named ``conftest.py`` so it does
not affect the existing integration test suite. New integration tests that
want to exercise a real Postgres and/or Redis must add::

    pytest_plugins = ["tests.integration.conftest_integration"]

at the top of the module, and then request the ``real_db_engine``,
``real_redis`` or ``api_client`` fixtures.

All fixtures are defensively gated: if the underlying services cannot be
reached (e.g. a local dev environment with no infra running), the fixtures
call ``pytest.skip`` so tests depending on them are skipped cleanly rather
than erroring. This means the suite can be committed and pushed without
breaking local runs.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import AsyncIterator, Iterator

import pytest


def _load_dev_env_file() -> None:
    """Populate os.environ from backend/env_config/synced/.env.$ENVIRONMENT.

    The backend container receives this file via compose's ``env_file:``. When
    running integration tests on the host, the same vars (TMDB_API_KEY,
    FIREBASE_*, DB_*, REDIS_URL, ...) are needed so the FastAPI app's
    AppSettings singleton can initialize. Only missing keys are filled — real
    shell env always wins.
    """

    env_name = os.environ.get("ENVIRONMENT", "dev")
    # backend/tests/integration/conftest_integration.py -> backend/
    backend_root = Path(__file__).resolve().parents[2]
    env_file = backend_root / "env_config" / "synced" / f".env.{env_name}"
    if not env_file.is_file():
        return

    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

    # Rewrite Docker-network hostnames to host-facing localhost. Compose maps
    # the service ports to the same numbers on localhost, so this "just works"
    # while `make dev-start` is running.
    _host_overrides = {
        "DB_HOST": "localhost",
        "REDIS_URL": "redis://localhost:6379",
        "NEO4J_URI": "bolt://localhost:7687",
    }
    for key, host_value in _host_overrides.items():
        current = os.environ.get(key, "")
        if "localhost" not in current and "127.0.0.1" not in current:
            os.environ[key] = host_value


_load_dev_env_file()

# ---------------------------------------------------------------------------
# Defaults mirror docker-compose.dev.yml so a local `make dev-start` "just works".
# ---------------------------------------------------------------------------

_DEFAULT_DATABASE_URL = (
    "postgresql+asyncpg://db-dev-user:db-dev-password@localhost:5432/dev-db"
)
_DEFAULT_REDIS_URL = "redis://localhost:6379/0"

_TEST_FIREBASE_UID = "integration-test-user"
_TEST_EMAIL = "integration-test@example.com"


def _database_url() -> str:
    return os.environ.get("DATABASE_URL", _DEFAULT_DATABASE_URL)


def _redis_url() -> str:
    return os.environ.get("REDIS_URL", _DEFAULT_REDIS_URL)


def _run(coro):
    """Run a coroutine on a fresh event loop.

    Avoids clashing with pytest-asyncio's per-test loop management when we
    need to do synchronous setup/teardown inside a fixture body.
    """

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Real Postgres engine
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_db_engine():
    """Async SQLAlchemy engine backed by a real Postgres.

    Skips the test if:
    - sqlalchemy/asyncpg aren't installed, or
    - the database can't be reached, or
    - the ``movie_recommender.database.models`` metadata can't be created.

    We create tables via ``metadata.create_all`` rather than Alembic so the
    fixture is self-contained and does not depend on the project Alembic
    environment (which reads settings on import).
    """

    pytest.importorskip("sqlalchemy")
    pytest.importorskip("sqlalchemy.ext.asyncio")
    pytest.importorskip("asyncpg")

    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import NullPool

        from movie_recommender.database.models import metadata
    except Exception as exc:  # pragma: no cover - defensive
        pytest.skip(f"integration services unavailable: cannot import models ({exc})")

    engine = create_async_engine(_database_url(), future=True, poolclass=NullPool)

    async def _setup() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def _teardown() -> None:
        try:
            async with engine.begin() as conn:
                await conn.run_sync(metadata.drop_all)
        finally:
            await engine.dispose()

    try:
        _run(_setup())
    except Exception as exc:
        pytest.skip(f"integration services unavailable: postgres ({exc})")

    yield engine

    try:
        _run(_teardown())
    except Exception:
        # Best effort — don't mask test failures with teardown errors.
        pass


# ---------------------------------------------------------------------------
# Real Redis
# ---------------------------------------------------------------------------


@pytest.fixture
async def real_redis() -> AsyncIterator["object"]:
    """redis.asyncio client connected to a real Redis.

    Skipped if the redis package is missing or the server is unreachable.
    Yields a client namespaced to a unique key prefix (available on the
    ``.test_prefix`` attribute) so parallel runs don't collide.
    """

    redis_pkg = pytest.importorskip("redis.asyncio")
    client = redis_pkg.from_url(_redis_url(), encoding="utf-8", decode_responses=True)

    try:
        await client.ping()
    except Exception as exc:
        try:
            await client.aclose()
        finally:
            pytest.skip(f"integration services unavailable: redis ({exc})")

    # Attach a per-test unique prefix so callers can scope their keys.
    client.test_prefix = f"test:integration:{uuid.uuid4().hex}:"

    try:
        yield client
    finally:
        # Best-effort cleanup of any keys under our prefix.
        try:
            async for key in client.scan_iter(match=f"{client.test_prefix}*"):
                await client.delete(key)
        except Exception:
            pass
        await client.aclose()


# ---------------------------------------------------------------------------
# FastAPI TestClient with real dependency graph + stubbed auth
# ---------------------------------------------------------------------------


@pytest.fixture
def api_client(real_db_engine) -> Iterator["object"]:
    """FastAPI TestClient wired to a real Postgres.

    - Overrides ``get_db`` to use the ``real_db_engine`` session factory so
      writes land in the real DB provisioned by the fixture above.
    - Stubs ``firebase_admin.auth.verify_id_token`` and ``auth.get_user`` so
      requests bypass real Firebase and resolve to a deterministic test user.
    - Provisions the test user row in ``users`` so endpoints that look up
      internal user ids by firebase uid succeed.

    Skipped if the FastAPI app module can't be imported (e.g. missing env
    vars for Firebase/Neo4j at import time).
    """

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    pytest.importorskip("fastapi.testclient")

    try:
        from fastapi.testclient import TestClient
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        from movie_recommender.dependencies.database import get_db
        from movie_recommender.main import app
    except Exception as exc:
        pytest.skip(f"integration services unavailable: fastapi app import ({exc})")

    session_factory = async_sessionmaker(
        real_db_engine, class_=AsyncSession, expire_on_commit=False
    )

    async def _override_get_db():
        async with session_factory() as session:
            try:
                yield session
            finally:
                await session.close()

    # Seed the test user so downstream lookups by firebase_uid succeed.
    from movie_recommender.database.models import users as users_table
    from sqlalchemy import insert, select

    async def _ensure_test_user() -> int:
        async with session_factory() as session:
            existing = await session.execute(
                select(users_table.c.id).where(
                    users_table.c.firebase_uid == _TEST_FIREBASE_UID
                )
            )
            row = existing.first()
            if row is not None:
                return row.id
            result = await session.execute(
                insert(users_table)
                .values(firebase_uid=_TEST_FIREBASE_UID, email=_TEST_EMAIL)
                .returning(users_table.c.id)
            )
            await session.commit()
            return result.scalar_one()

    try:
        user_id = _run(_ensure_test_user())
    except Exception as exc:
        pytest.skip(f"integration services unavailable: db seed ({exc})")

    app.dependency_overrides[get_db] = _override_get_db

    # Stub firebase verification so `verify_user()` returns our test user.
    # We monkey-patch at the firebase_admin layer rather than overriding the
    # dependency key, because `verify_user()` is a *factory* invoked at route
    # import time — each call yields a unique closure, making it impossible
    # to address via `dependency_overrides`.
    import firebase_admin.auth as fb_auth

    class _FakeUserRecord:
        email_verified = True
        display_name = "Integration Test"

    original_verify = getattr(fb_auth, "verify_id_token", None)
    original_get_user = getattr(fb_auth, "get_user", None)

    def _fake_verify_id_token(_token, *_a, **_kw):
        return {"uid": _TEST_FIREBASE_UID, "email": _TEST_EMAIL}

    def _fake_get_user(_uid, *_a, **_kw):
        return _FakeUserRecord()

    fb_auth.verify_id_token = _fake_verify_id_token
    fb_auth.get_user = _fake_get_user

    try:
        # raise_server_exceptions=False lets the test suite assert on HTTP
        # contract (status codes / bodies) even when downstream dependencies
        # like the recommender artifacts or Neo4j aren't wired up in CI —
        # otherwise a FileNotFoundError in a route handler re-raises through
        # TestClient and the test dies before it can check the response.
        client = TestClient(app, raise_server_exceptions=False)
        # Attach the internal user id for tests that need it.
        client.test_user_id = user_id
        client.test_firebase_uid = _TEST_FIREBASE_UID
        # Default auth header so routes pass the OAuth2 scheme.
        client.headers.update({"Authorization": "Bearer fake-integration-token"})
        yield client
    finally:
        if original_verify is not None:
            fb_auth.verify_id_token = original_verify
        if original_get_user is not None:
            fb_auth.get_user = original_get_user
        app.dependency_overrides.clear()
