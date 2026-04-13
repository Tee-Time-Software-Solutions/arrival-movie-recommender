# Roadmap

## Milestone 1: Backend Testing

### Phase 1: ML Pipeline Testing
- Comprehensive unit and integration test coverage for all backend modules
- API endpoint testing with FastAPI TestClient
- Test consolidation and cleanup

### Phase 2: Additive CI Expansion

**Goal:** Additively expand CI coverage — backend-first, frontend deferred.
**Depends on:** Phase 1
**Plans:** 5 plans (4 backend-priority + 1 deferred frontend)

Plans:
- [ ] 02-01 — Backend unit test expansion (Wave 1, priority)
- [ ] 02-02 — Backend integration tests (Wave 1, priority)
- [ ] 02-03 — Integration-test CI workflow (Wave 1, priority)
- [ ] 02-05 — Backend coverage CI workflow (Wave 1, priority)
- [ ] 02-04 — Frontend test bootstrap + ci-frontend.yml **(DEFERRED — optional, only if time permits after 01/02/03/05)**

**Priority note:** Frontend tests are deprioritized. Phase 2 is considered complete once 02-01, 02-02, 02-03, and 02-05 are done. 02-04 is a stretch/bonus plan.

**Details:**
Expand CI coverage additively — no edits to existing source, tests, workflows, Makefile, or pre-commit. New files only under `backend/tests/`, frontend test dirs, and new `.github/workflows/*.yml`. Scope: (1) new backend unit tests for uncovered modules, (2) new backend integration tests, (3) new integration-test workflow with Postgres/Redis service containers, (4) backend coverage reporting workflow. Deferred: frontend test suite from scratch (Vitest + RTL) + frontend CI workflow + frontend coverage — bundled in Plan 02-04.
