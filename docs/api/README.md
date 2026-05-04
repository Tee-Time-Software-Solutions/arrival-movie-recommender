# API Documentation

Reference for the Arrival movie recommender REST + SSE API. This complements (does not replace) the **auto-generated Swagger UI** that ships with the backend. Use Swagger to play with payloads; use these docs to learn the layout, conventions, and how to add an endpoint.

> **Base URL:** `/api/v1` (mounted in [backend/src/movie_recommender/main.py:66](../../backend/src/movie_recommender/main.py#L66))
> **Swagger UI:** `http://localhost:8000/docs`
> **ReDoc:** `http://localhost:8000/redoc`
> **OpenAPI JSON:** `http://localhost:8000/openapi.json`

---

## Endpoint reference

One file per router. Each page covers paths, payloads, auth requirements, and gotchas for that domain.

| Domain | File | Routes |
|---|---|---|
| Health | [health.md](health.md) | Liveness + dependency checks |
| Chatbot (SSE) | [chatbot.md](chatbot.md) | LangGraph agent streaming endpoint |
| Movies | [movies.md](movies.md) | Recommendation feed (Redis-backed queue) |
| Interactions | [interactions.md](interactions.md) | Swipes (like / dislike / skip) |
| Users | [users.md](users.md) | Profile, preferences, registered users |
| Onboarding | [onboarding.md](onboarding.md) | First-run flow + TMDB search |
| People | [people.md](people.md) | Knowledge-graph person lookups |
| Watchlist | [watchlist.md](watchlist.md) | Save-for-later list |

For deeper agent internals, see [../chatbot_agent/langgraph_agent.md](../chatbot_agent/langgraph_agent.md).

---

## Quickstart

```bash
# 1. Start the dev environment (Postgres, Redis, Neo4j, backend, frontend)
make dev-start

# 2. Generate a Firebase ID token for a test user
make gen-dev-token
# → prints: eyJhbGciOi...

# 3. Hit an authenticated endpoint
curl -H "Authorization: Bearer <PASTE_TOKEN>" \
     http://localhost:8000/api/v1/health/ping

# Response: {"response":"pong"}
```

Open `http://localhost:8000/docs` and click **Authorize** to paste the same token — you can then call any endpoint from the browser.

---

## Authentication

Every endpoint except `/health/*` requires a **Firebase ID token** in the `Authorization: Bearer <token>` header.

The dependency that enforces this is `verify_user(...)` in [backend/src/movie_recommender/dependencies/firebase.py](../../backend/src/movie_recommender/dependencies/firebase.py):

| Flag | Effect |
|---|---|
| `verify_user()` | Token must be valid. The decoded token is returned to the route. |
| `verify_user(email_needs_verification=True)` | Also requires `email_verified=True` (currently unused on any route). |
| `verify_user(user_private_route=True)` | Used on `/users/{user_id}/...` routes. The path's `{user_id}` must equal the token's `uid` — prevents user A from reading user B's data. |

Validation chain on every authenticated call:

1. Extract token from `Authorization` header.
2. `firebase_admin.auth.verify_id_token(token)` → decoded claims (`uid`, `email`, ...).
3. `firebase_admin.auth.get_user(uid)` → user record (used for `email_verified`, `display_name`).
4. (If `user_private_route`) compare `decoded["uid"]` to path `{user_id}`.
5. Return decoded token + user record fields to the handler as `auth_user`.
6. The handler typically calls `get_user_by_firebase_uid(db, auth_user["uid"])` to load the **internal** numeric `user_id` from Postgres.

Failure cases:

- Missing/invalid token → **401** `{"detail": "Authentication failed"}`
- Email-verification fail → **403** `{"detail": "Email not verified"}`
- Ownership fail → **403** `{"detail": "Access denied: you can only access your own resources"}`

---

## Error codes

The API uses standard FastAPI error responses. All non-streaming errors come back as JSON `{"detail": "<message>"}` with the appropriate status code.

| Status | Meaning | Common causes |
|---|---|---|
| 400 | Bad request | Invalid parameter combination (e.g. `skip` + `supercharged`) |
| 401 | Unauthorized | Missing/expired/invalid Firebase token |
| 403 | Forbidden | Email not verified, or accessing another user's `/users/{user_id}/*` |
| 404 | Not found | User, movie, or watchlist entry doesn't exist |
| 409 | Conflict | Resource already exists (registered user, watchlist entry, completed onboarding) |
| 422 | Unprocessable entity | Pydantic validation failed — payload shape mismatch |
| 502 | Bad gateway | Upstream TMDB call failed |
| 503 | Service unavailable | Onboarding seed missing, or `/health/dependencies` reports a down dependency |
| 500 | Server error | Anything else — reported in logs |

For SSE streams, transport-level errors are surfaced as an `event: error` frame instead of an HTTP status. The HTTP status will still be **200** — that's the streaming response itself succeeding.

---

## How to add an endpoint

Mirror the existing layered architecture: **Router → Dependency → CRUD → DB**. Concrete steps using a fictional `notes` domain.

### Step 1 — Pydantic schemas

`backend/src/movie_recommender/schemas/requests/notes.py`:
```python
from pydantic import BaseModel

class NoteCreate(BaseModel):
    content: str

class NoteResponse(BaseModel):
    id: int
    content: str
```

If you also want a typed view of the DB row, add a `TypedDict` in `schemas/database/notes.py`.

### Step 2 — CRUD

`backend/src/movie_recommender/database/CRUD/notes.py`. **Async, SQLAlchemy Core, no ORM.**

```python
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.models import notes

async def create_note(db: AsyncSession, user_id: int, content: str) -> int:
    result = await db.execute(
        insert(notes).values(user_id=user_id, content=content).returning(notes.c.id)
    )
    await db.commit()
    return result.scalar_one()
```

If a new table is needed, add it to `database/models.py` and create an Alembic migration:
```bash
cd backend && alembic revision --autogenerate -m "add notes table" && alembic upgrade head
```

### Step 3 — Router

`backend/src/movie_recommender/api/v1/notes.py`:
```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.CRUD.notes import create_note
from movie_recommender.database.CRUD.users import get_user_by_firebase_uid
from movie_recommender.dependencies.database import get_db
from movie_recommender.dependencies.firebase import verify_user
from movie_recommender.schemas.requests.notes import NoteCreate, NoteResponse

router = APIRouter(prefix="/notes", tags=["notes"])

@router.post("")
async def create(
    body: NoteCreate,
    db: AsyncSession = Depends(get_db),
    auth_user=Depends(verify_user()),
) -> NoteResponse:
    user = await get_user_by_firebase_uid(db, auth_user["uid"])
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    note_id = await create_note(db, user.id, body.content)
    return NoteResponse(id=note_id, content=body.content)
```

### Step 4 — Register the router

`backend/src/movie_recommender/api/v1/__init__.py`:
```python
from .notes import router as notes_router
# ...
routers = [
    chatbot_router,
    health_router,
    # ...
    notes_router,  # ← new
]
```

The main app picks this up automatically: `app.include_router(router, prefix="/api/v1")` ([main.py:66](../../backend/src/movie_recommender/main.py#L66)).

### Step 5 — Tests

Two layers, mirroring [backend/tests/](../../backend/tests/):

**Unit test** — `backend/tests/unit/api/test_notes.py`. Use `AsyncMock` for the DB session, override FastAPI dependencies via `app.dependency_overrides[get_db]`, patch CRUD functions at the import site:

```python
from unittest.mock import AsyncMock, patch
# ...
with patch("movie_recommender.api.v1.notes.create_note", AsyncMock(return_value=1)):
    resp = client.post("/api/v1/notes", json={"content": "hi"})
    assert resp.status_code == 200
```

**Integration test** — `backend/tests/integration/test_notes.py`. Use the existing `api_client` fixture (real Postgres, stubbed Firebase). Mark with `pytest.mark.integration`. See [backend/tests/integration/test_api_smoke.py](../../backend/tests/integration/test_api_smoke.py) for the pattern.

### Step 6 — Frontend wiring (if user-facing)

1. Type in `frontend/app/src/types/notes.ts`.
2. API call in `frontend/app/src/services/api/notes.ts` using the shared `apiClient` (Firebase token auto-attached).
3. (Optional) Zustand store in `frontend/app/src/stores/notesStore.ts`.
4. Page/feature component under `frontend/app/src/app/` or `components/features/`.

---

## Local dev cheatsheet

```bash
make dev-start            # Start everything (Postgres, Redis, Neo4j, backend, frontend)
make dev-stop             # Stop everything
make dev-rebuild          # Rebuild containers after code changes
make backend-tests        # Backend unit tests (no external services needed)
make format-check         # Lint
make gen-dev-token        # Dev Firebase token for curl/Postman/Swagger

# Backend integration tests (requires `make dev-start` first)
cd backend && .venv/bin/python -m pytest tests/integration/ -v -m integration

# Apply migrations
cd backend && alembic upgrade head
```

---

## CI overview

Three GitHub Actions workflows trigger on backend changes (`backend/**`):

| Workflow | File | Trigger | What it does |
|---|---|---|---|
| **CI - Backend** | [.github/workflows/ci-backend.yml](../../.github/workflows/ci-backend.yml) | push / PR on any branch | Ruff lint → unit tests (`unit-tests` needs `lint`) |
| **CI - Backend Integration** | [.github/workflows/ci-backend-integration.yml](../../.github/workflows/ci-backend-integration.yml) | push / PR on any branch | Boots Postgres 16 + Redis 7.2 service containers, runs Alembic migrations, runs `pytest -m integration`. Stub values for Firebase/Neo4j/TMDB so `AppSettings` validates without real credentials |
| **Coverage Report** | [.github/workflows/coverage-report.yml](../../.github/workflows/coverage-report.yml) | push to `main` + PRs | Unit tests with `pytest-cov`, uploads `coverage.xml` artifact |

A frontend-related lane (`ci-infra.yml`) covers infra/deployment validation. PRs that touch only `frontend/**` typically do not trigger backend lanes — check the workflow `paths:` filters before assuming green CI.

---

## Conventions worth knowing

- **Async everywhere.** All I/O (DB, Redis, HTTP) is async. Don't add sync calls in handlers.
- **SQLAlchemy Core, not ORM.** Tables in `database/models.py` use `Table()` + `Column()`. Queries use `select()`, `insert()`, `update()`. No mapped classes.
- **Singletons** — `AppSettings`, `DatabaseEngine`, `RedisClient` use `__new__` for single instances. Inject via `Depends(...)` rather than calling `AppSettings()` inside handlers.
- **Fire-and-forget** — see `interactions.py`: swipe persistence and beacon updates are wrapped in `asyncio.create_task(...)` so the request doesn't wait. Be careful: failures are logged, not surfaced to the client.
- **Pydantic for I/O, TypedDicts for DB rows.** Don't return raw SQLAlchemy `Row` objects from CRUD; project them into either a Pydantic model or a `TypedDict` in `schemas/database/`.
- **Tag your routers** (`tags=["notes"]`) so Swagger groups them.
- **Owner-scoped routes** use `verify_user(user_private_route=True)`. The path *must* contain `{user_id}` or it'll **500**.
