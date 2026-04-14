# CLAUDE.md — Arrival Movie Recommender

## Project Overview

A movie recommendation app with a Tinder-style swipe interface. Users swipe on movies (like/dislike/skip), and a recommendation engine learns their preferences.

- **Backend**: FastAPI (Python 3.11), async-first, SQLAlchemy Core, PostgreSQL, Redis
- **Frontend**: React 19 + TypeScript, Vite, Zustand, Framer Motion, Tailwind CSS v4
- **Auth**: Firebase (Google OAuth + email/password)
- **Containerization**: Docker Compose (dev), PostgreSQL 16, Redis 7.2

---

## Repository Structure

```
├── backend/
│   └── src/movie_recommender/
│       ├── api/v1/              # FastAPI routers (one per domain)
│       ├── core/                # Infrastructure (settings, logger, clients)
│       ├── database/
│       │   ├── models.py        # SQLAlchemy table definitions
│       │   ├── engine.py        # Async DB engine (singleton)
│       │   ├── CRUD/            # Data access functions (one file per entity)
│       │   └── migrations/      # Alembic migrations
│       ├── dependencies/        # FastAPI Depends() factories
│       ├── schemas/
│       │   ├── requests/        # Pydantic request/response models
│       │   └── database/        # DB row TypedDicts
│       └── services/            # Business logic (feed_manager, hydrator, recommender)
├── frontend/app/src/
│   ├── app/                     # Page components (one folder per page)
│   ├── components/
│   │   ├── ui/                  # Shell/layout components (Layout, Sidebar, BottomNav)
│   │   └── features/           # Feature components (SwipeDeck, MovieCard, MovieDetail)
│   ├── stores/                  # Zustand stores (auth, movie, chat)
│   ├── services/api/           # Axios API layer (client + endpoint modules)
│   ├── hooks/                   # Custom React hooks (useAuth, useChat)
│   ├── types/                   # TypeScript interfaces (one file per domain)
│   ├── lib/                     # Utilities (firebase init, cn(), constants)
│   └── core/                    # App config
├── deployment/                  # Docker Compose files
└── makefile                     # Dev commands (install, dev-start, tests, etc.)
```

---

## Backend Patterns

### Layered Architecture

Requests flow through: **Router → Dependencies → CRUD/Service → Database**

1. **Routers** (`api/v1/`): Thin HTTP handlers. Validate input via Pydantic, call CRUD/service functions, return responses. Each file covers one domain (users, movies, interactions, health).

2. **Dependencies** (`dependencies/`): FastAPI `Depends()` factories that provide request-scoped resources (DB session, auth token, feed manager). Used in router function signatures.

3. **CRUD** (`database/CRUD/`): Pure async functions for database operations. Use SQLAlchemy Core (NOT ORM). Return TypedDicts for type safety. One file per entity.

4. **Services** (`services/`): Business logic that orchestrates CRUD + external APIs. Examples: `FeedManager` (Redis queue management), `MovieHydrator` (TMDB enrichment), `Recommender` (ML engine).

### Key Conventions

- **Singletons**: `AppSettings`, `DatabaseEngine`, `RedisClient` use `__new__()` pattern for single instances.
- **Async everywhere**: All I/O is async (asyncpg, redis.asyncio, httpx).
- **SQLAlchemy Core, not ORM**: Tables defined with `Table()` and `Column()` in `models.py`. Queries use `select()`, `insert()`, `update()` — no mapped classes.
- **TypedDicts for DB rows**: Define expected shapes of query results in `schemas/database/` for IDE autocompletion.
- **Pydantic for I/O**: All request bodies, response bodies, and settings use Pydantic models in `schemas/requests/`.
- **Auth via dependency**: `verify_user()` returns a dependency function. Use `user_private_route=True` for ownership checks.

### Adding a New Backend Endpoint

1. Define Pydantic request/response models in `schemas/requests/<domain>.py`
2. Write CRUD functions in `database/CRUD/<domain>.py` (async, SQLAlchemy Core)
3. Add the route in `api/v1/<domain>.py` using dependency injection
4. If new domain, create a new router file and register it in `api/v1/__init__.py`

### API URL Pattern

All routes are prefixed with `/api/v1/`. Router registration happens in `api/v1/__init__.py` with tag-based grouping.

### Database Migrations

```bash
alembic revision --autogenerate -m "description"
alembic upgrade head
```

Migration env loads connection URL from `AppSettings` singleton.

### Testing

#### Test structure

- **Unit tests**: `backend/tests/unit/`, mirroring source layout (`api/`, `services/`, `database/CRUD/`, etc.)
- **Integration tests**: `backend/tests/integration/`, marked with `@pytest.mark.integration`
- Fixtures in `conftest.py` files — global root conftest plus per-directory overrides
- Mock async dependencies with `AsyncMock`
- Patch at the import site: `movie_recommender.api.v1.users.create_user`
- Override FastAPI dependencies via `app.dependency_overrides[get_db]`

#### Running locally

```bash
make backend-tests          # unit tests only (no external services needed)

# Integration tests require Postgres + Redis running (e.g. via make dev-start):
cd backend
.venv/bin/python -m pytest tests/integration/ -v -m integration
```

#### CI pipelines (GitHub Actions)

Three workflows trigger on backend changes (`backend/**`):

| Workflow | File | Trigger | What it does |
|---|---|---|---|
| **CI - Backend** | `ci-backend.yml` | push / PR on any branch | Lint (Ruff) → unit tests, sequentially (`unit-tests` needs `lint`) |
| **CI - Backend Integration** | `ci-backend-integration.yml` | push / PR on any branch | Spins up Postgres 16 + Redis 7.2 as services, runs Alembic migrations, then runs `pytest tests/integration/ -m integration` |
| **Coverage Report** | `coverage-report.yml` | push to `main` + PRs | Runs unit tests with `pytest-cov`, uploads `coverage.xml` as a build artifact |

**Integration workflow environment** — the workflow injects all required `AppSettings` fields as env vars so the app can boot without real Firebase/Neo4j/TMDB credentials. Stub values are used for those services; only Postgres and Redis are real. Key vars set:

- `DATABASE_URL`, `DB_*` — points to the GitHub-hosted Postgres service container
- `REDIS_URL` — points to the GitHub-hosted Redis service container
- `FIREBASE_*`, `NEO4J_*`, `TMDB_*` — stub/fake values so `AppSettings` validation passes without live credentials

---

## Frontend Patterns

### Component Organization

- **Pages** (`app/`): One folder per route. Page components handle data fetching, state orchestration, and layout composition. Examples: `DiscoverPage`, `ProfilePage`, `ChatPage`, `LandingPage`.
- **Feature components** (`components/features/`): Reusable domain-specific UI (SwipeDeck, MovieCard, MovieDetail, ChatMessage). Each in its own folder.
- **UI components** (`components/ui/`): Shell and layout primitives (Layout, Sidebar, BottomNav, ProtectedRoute). Includes shadcn/ui components.

### State Management (Zustand)

Three stores, one per domain:

- **authStore**: Firebase user, JWT token, loading state
- **movieStore**: Movie queue, current index, liked/disliked lists, watched movies
- **chatStore**: Messages array, typing indicator

Pattern: `create<StoreType>()((set, get) => ({ ... }))`. State updates are immutable with spread operators.

### API Layer

- **`services/api/client.ts`**: Axios instance with request interceptor that attaches Firebase JWT token.
- **`services/api/<domain>.ts`**: One file per domain. Functions call `apiClient.get/post/patch` and return typed responses.
- **Base URL**: Set via `VITE_BASE_URL` env variable.

### Type Definitions

- One file per domain in `types/` (movie.ts, user.ts, chat.ts)
- Interfaces mirror backend Pydantic schemas (snake_case field names)
- Frontend types and backend schemas must stay in sync manually

### Styling

- Tailwind CSS v4 with inline classes (no CSS modules)
- Theme defined via CSS variables in `index.css` using `@theme inline`
- `cn()` utility from `lib/utils.ts` (clsx + tailwind-merge) for conditional classes
- Framer Motion for animations and gesture handling

### Adding a New Frontend Feature

1. Define TypeScript interfaces in `types/<domain>.ts`
2. Add API functions in `services/api/<domain>.ts`
3. Create or extend a Zustand store in `stores/<domain>Store.ts`
4. Build feature component in `components/features/<FeatureName>/`
5. Create page component in `app/<PageName>/`
6. Add route in `App.tsx`

### UX Patterns in Use

- **Optimistic updates**: Swipe actions update UI immediately, API calls fire-and-forget
- **Queue prefetching**: Auto-fetch more movies when queue drops below threshold (3 remaining)
- **Gesture-first**: Framer Motion drag for swipes, long-press for supercharge, keyboard fallbacks
- **Responsive nav**: Desktop sidebar (hover-expand) + mobile bottom nav

---

## Development Commands

```bash
make install          # Install all dependencies (backend + frontend)
make dev-start        # Start dev environment (Docker Compose)
make dev-rebuild      # Rebuild and start
make dev-stop         # Stop dev environment
make backend-tests    # Run unit tests
make format-check     # Lint check
make gen-dev-token    # Generate Firebase token for local testing
```

Requires `ENVIRONMENT=dev` or `ENVIRONMENT=production`.

---

## Code Quality

- **Backend**: Ruff for linting/formatting, pre-commit hooks run tests
- **Frontend**: ESLint with React plugins
- **Pre-commit**: Trailing whitespace, YAML/JSON validation, large file detection, debug statement detection, gitleaks secret scanning
- **CI**: Three GitHub Actions workflows — lint+unit tests, integration tests (real Postgres/Redis), and coverage report (see Testing section)

---

## Key Architectural Decisions

- **SQLAlchemy Core over ORM**: Explicit queries, no magic, better async support
- **Zustand over Context**: Simpler API, better perf for frequent state updates
- **Vite over Next.js**: Client-side SPA only, no SSR needed
- **Firebase auth**: Outsourced auth, backend verifies ID tokens via dependency injection
- **Redis for feed queue**: Movies pre-fetched into per-user Redis queues, refilled async from recommender
- **TMDB hydration**: Movie stubs created first, then enriched with TMDB metadata on-demand
