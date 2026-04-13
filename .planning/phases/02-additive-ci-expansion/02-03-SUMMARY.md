---
phase: 02-additive-ci-expansion
plan: 03
status: complete
completed_at: 2026-04-13
---

# Plan 02-03 Summary: Additive Backend Integration CI Workflow

## Objective
Add a new GitHub Actions workflow that runs backend integration tests against real Postgres 16 + Redis 7.2 service containers, without editing any existing workflow, Makefile, or source.

## Outcome
One new workflow file created. Zero edits to existing files. YAML validates.

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Create ci-backend-integration.yml workflow | `18aa637` | `.github/workflows/ci-backend-integration.yml` |

## Implementation Notes

### Triggers
- `push` and `pull_request` on all branches.
- Path-scoped to `backend/**` and the workflow file itself (mirrors ci-backend.yml).

### Service Containers
- `postgres:16` with env `POSTGRES_USER=db-dev-user`, `POSTGRES_PASSWORD=db-dev-password`, `POSTGRES_DB=dev-db` (matches `deployment/docker-compose.dev.yml`). Health check via `pg_isready`.
- `redis:7.2` with health check via `redis-cli ping`.

### Steps
1. `actions/checkout@v4`
2. `astral-sh/setup-uv@v3` with cache (mirrors ci-backend.yml).
3. `actions/setup-python@v5` with Python 3.11.
4. `make install`
5. `.venv/bin/alembic upgrade head` — runs migrations against the postgres service.
6. `.venv/bin/python -m pytest tests/integration/ -v -m integration` — runs Plan 02-02's marked suite (API smoke + FeedManager Redis).
7. `.venv/bin/python -m pytest tests/integration/test_full_pipeline.py tests/integration/test_online_recommender.py -v` — runs the pre-existing unmarked ML pipeline tests so Phase 1 coverage is not regressed.

### Env Vars Stubbed
`AppSettings._initialize()` invokes `check_required()` for database + firebase and builds `TMDBSettings` unconditionally. The workflow sets all required env vars:
- DB_* (real values pointing at the postgres service) + DATABASE_URL.
- REDIS_URL, REDIS_MAX_CONNECTIONS.
- ENVIRONMENT=dev, BATCH_SIZE, QUEUE_MIN_CAPACITY.
- TMDB_API_KEY / TMDB_IMG_URL / TMDB_BASE_URL (fake values — integration tests do not call TMDB).
- NEO4J_* (fake — Neo4j is not a service in this workflow; Phase 2 integration tests don't depend on it).
- FIREBASE_* (fake — auth dependency is overridden in the test fixtures per Plan 02-02).

### Constraints Honored
- No changes to `backend/Makefile`.
- No edits to `.github/workflows/ci-backend.yml` (git diff empty).
- No edits to source code. AppSettings stays unchanged; fake env values satisfy `check_required`.
- Additive only: single new file.

## Verification
- `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci-backend-integration.yml'))"` -> `valid yaml`.
- `git diff .github/workflows/ci-backend.yml` -> empty.
- No changes to backend/Makefile.
- Workflow references `postgres:16` and `redis:7.2`.

## Dependencies / Assumptions
- Relies on Plan 02-02 having marked its new integration tests with `pytestmark = pytest.mark.integration`. If 02-02 has not yet landed on the branch, the `-m integration` step will report "no tests ran" and exit non-zero under default pytest config — this is acceptable because 02-02 and 02-03 are both in Wave 1 and are expected to merge together.
- Legacy `test_full_pipeline.py` and `test_online_recommender.py` download the MovieLens 20M dataset (~190MB) and run a ~5min pipeline on first run; CI runner has enough disk/compute. No caching added in this plan (future work).

## Files Modified
- `.github/workflows/ci-backend-integration.yml` (new)

## Files Explicitly NOT Modified
- `.github/workflows/ci-backend.yml`
- `backend/Makefile`
- Any source or test files
