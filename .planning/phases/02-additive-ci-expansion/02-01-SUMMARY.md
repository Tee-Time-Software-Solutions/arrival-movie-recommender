# Plan 02-01 Summary: Additive Backend Unit Test Expansion

## Status
Complete. All tasks executed, all tests passing, zero edits to existing files.

## Goal
Expand backend unit test coverage additively — five new test files under
`backend/tests/unit/` covering previously uncovered high-value modules:
FeedManager, swipe_worker, and the api/{health, movies, watchlist} routers.

## Deliverables

### Task 1: Service-layer unit tests
- `backend/tests/unit/services/feed_manager/__init__.py` (new)
- `backend/tests/unit/services/feed_manager/test_feed_manager.py` (new) — 15 tests
  - `TestFlushFeed` (2): per-user queue key deletion
  - `TestPopUnseen` (4): empty-queue, first-unseen, seen-skip, all-seen-exhaustion
  - `TestGetNextMovie` (6): happy path, seen-set dedup from Redis, empty-queue refill,
    no-movies-after-refill, background refill below threshold, no refill when healthy
  - `TestRefillQueue` (3): queue population + cached seen invalidation, batch_size cap,
    skipped failed hydrations
- `backend/tests/unit/services/swipe_worker/__init__.py` (new)
- `backend/tests/unit/services/swipe_worker/test_swipe_worker.py` (new) — 6 tests
  - `TestEnqueueSwipe` (2): JSON payload shape, supercharged flag propagation
  - `TestDrainSwipeQueue` (4): batched persistence + cancel, batch_size boundary,
    malformed event survives worker, no DB session opened on empty queue

**Commit:** `c204004 test(02-01): unit tests for FeedManager and swipe_worker services`

### Task 2: API router unit tests
- `backend/tests/unit/api/test_health.py` (new) — 5 tests
  - `TestPing` (2): pong response, no auth required
  - `TestCheckDependencies` (3): all healthy 200, redis-down 503, neo4j-down 503
- `backend/tests/unit/api/test_movies.py` (new) — 8 tests
  - `TestFetchMoviesFeed` (3): hydrated movie return, 404 when no movies,
    404 when user not found
  - `TestFetchMoviesFeedBatch` (3): count honored, 404 when empty, 422 when count > 20
  - `TestFlushMoviesFeed` (2): flush success, 404 when user not found
- `backend/tests/unit/api/test_watchlist.py` (new) — 10 tests
  - `TestAddMovieToWatchlist` (4): success, user-not-found, movie-not-found, duplicate 409
  - `TestRemoveMovieFromWatchlist` (3): success, user-not-found, not-in-watchlist 404
  - `TestGetWatchlistMovies` (3): paginated response with ownership scope check,
    user-not-found 404, empty watchlist

**Commit:** `312d10d test(02-01): unit tests for api/health, api/movies, api/watchlist routers`

## Test counts
- **Added:** 44 new unit tests (15 + 6 + 5 + 8 + 10)
- **Backend unit suite before:** ~183 passing (STATE.md baseline 122 + other wave 1 work)
- **Backend unit suite after:** 227 passing

## Verification
- `cd backend && .venv/bin/python -m pytest tests/unit -q` → 227 passed, 5 warnings
- `.venv/bin/python -m ruff check <new files>` → all checks passed
- `git diff --name-only HEAD~2 HEAD` shows only new files (7 additions, 0 modifications)

## Patterns / decisions
- **FeedManager construction in tests** bypasses `__init__` (which calls
  `AppSettings()` at module import time) via `FeedManager.__new__(FeedManager)`
  with a `SimpleNamespace` stub for `self.settings`. This matches the pattern
  used for `Recommender` in `tests/conftest.py`.
- **swipe_worker drain loop** is an infinite `while True`. Tests launch it as
  an asyncio Task, let it process a batch, then cancel and assert on the
  mocked `create_swipe` call count. `create_swipe` is patched at the import
  site (`movie_recommender.services.swipe_worker.main.create_swipe`).
- **API router tests** reuse the shared `client` fixture from
  `tests/unit/api/conftest.py` (no edits). The movies tests override
  `get_feed_manager` via `app.dependency_overrides` — the same pattern
  `test_interactions.py` uses for `get_recommender`/`get_async_redis`.
- **Neo4j mocking** for health tests: hand-rolled async-context-manager class
  instead of `AsyncMock` because the router code uses
  `async with neo4j_driver.session()` — `.session()` itself is synchronous,
  only the context-manager protocol is async.

## No files modified
Verified via `git diff --name-only HEAD~2 HEAD` — only the 7 new files
appear. No conftest, router, Makefile, or existing test was touched.
