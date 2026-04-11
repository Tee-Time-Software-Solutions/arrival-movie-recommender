# Roadmap — Arrival Movie Recommender

## Evaluation Rubric

| Criteria | Weight | Current Status |
|----------|--------|----------------|
| Recommender / Chatbot | **5/10** | Recommender: strong. Chatbot: echo-only stub |
| EDA | **1/10** | **Done** — EDA.ipynb merged (PR #75) |
| Deployment | **1/10** | Docker Compose local only, no cloud |
| MLOps | **1/10** | Reproducible pipeline, but no tracking/versioning |
| Dev Practices | **1/10** | **Done** — CI, PRs, CODEOWNERS, pre-commit, branches |

---

## What We Already Have

### Recommender (solid)

- Implicit ALS collaborative filtering (64-dim embeddings, 13K movies, 138K users)
- Full offline pipeline: preprocessing → chronological split → training → evaluation (Recall@10=0.08, Precision@10=0.05, NDCG@10=0.18)
- Online learning with gradient updates on new swipes (η=0.05, norm cap=10)
- Knowledge graph in Neo4j (beacon profiling, entity-weighted user preferences, explainable recs)
- Redis-backed feed queue with async refill
- Real-time serving via dot-product ranking

### Chatbot (stub only)

- Frontend shell: `ChatPage.tsx`, `chatStore.ts`, `chat.ts`
- Currently echo mode — returns `Echo: ${message}` with a 500ms delay
- No backend endpoint, no LLM, no recommendation-aware conversation

### EDA (done)

- `EDA.ipynb` merged via PR #75
- Dataset health, rating distribution, user/item activity histograms, Lorenz curve, temporal analysis
- Uses pandas, matplotlib, seaborn

### Deployment

- Docker Compose with PostgreSQL 16, Redis 7.2, Neo4j 5, backend, frontend
- Backend Dockerfile (Python 3.11-slim, Gunicorn + Uvicorn)
- Makefile automation (`make dev-start`, `dev-rebuild`, `dev-stop`)
- No cloud deployment

### MLOps

- Reproducible offline pipeline (`offline_pipeline.py`)
- Artifacts saved as `.npy` + `.json` files
- No experiment tracking (MLflow/W&B), no model registry, no automated retraining

### Dev Practices (done)

- 75+ merged PRs, 128+ commits, 8 contributors
- Feature branches with naming convention (`feat/*`, `chore/*`, `data/*`)
- CODEOWNERS with domain ownership
- Pre-commit hooks (Ruff, gitleaks, tests)
- GitHub Actions CI (lint → unit tests)
- 122 unit tests + 74 integration tests
- Alembic migrations

---

## Remaining Work — Prioritized

### PRIORITY 1: Chatbot Backend (Critical — part of the 5/10 bucket)

The single biggest gap. The rubric says "recommender/**chatbot**" and we have no chatbot.

| Task | Owner | Effort | Branch |
|------|-------|--------|--------|
| **1a.** Create backend chat router at `api/v1/chat.py` with `POST /api/v1/chat/message` | Backend | ~2h | `feat/chatbot-backend` |
| **1b.** Create chat service that calls Claude/Gemini API with system prompt including user's beacon map + recent swipe history | Backend | ~3h | same |
| **1c.** Wire the chat service to the recommender — chatbot can ask "what kind of movies do you like?" and feed preferences into the recommendation engine | Backend | ~2h | same |
| **1d.** Replace echo mode in `chat.ts` with real API calls | Frontend | ~1h | `feat/chatbot-frontend` |
| **1e.** Add conversation context — chatbot knows what movies the user has liked/disliked and can explain recommendations | Backend | ~2h | same |

**Deliverable:** A working chatbot that can discuss movies, explain recommendations, and influence the feed.

---

### PRIORITY 2: Deployment (1/10 bucket)

We have Docker Compose locally. We need a real deployment.

| Task | Owner | Effort | Branch |
|------|-------|--------|--------|
| **2a.** Deploy to a cloud provider — simplest: **Railway**, **Render**, or **Fly.io** (free tiers) | DevOps | ~3h | `feat/deployment` |
| **2b.** Add a production `docker-compose.prod.yml` with proper env vars, no debug mode, HTTPS | DevOps | ~1h | same |
| **2c.** Add deployment instructions to README | DevOps | ~30min | same |
| **2d.** (Bonus) Add a GitHub Actions deploy workflow triggered on merge to `main` | DevOps | ~2h | same |

**Deliverable:** App accessible at a public URL with documented deployment process.

---

### PRIORITY 3: MLOps (1/10 bucket)

| Task | Owner | Effort | Branch |
|------|-------|--------|--------|
| **3a.** Add **MLflow** tracking to `offline_pipeline.py` — log hyperparams, metrics (Recall/Precision/NDCG), and artifacts | ML team | ~2h | `feat/mlops` |
| **3b.** Add `mlflow ui` to Docker Compose as a service | ML team | ~1h | same |
| **3c.** Version model artifacts — either MLflow model registry or DVC | ML team | ~2h | same |
| **3d.** (Bonus) Retraining script triggered by cron or manual GitHub Action | ML team | ~2h | same |

**Deliverable:** MLflow dashboard showing experiment history, model metrics, and versioned artifacts.

---

## Parallel Execution Plan

```
Week 1 (parallel tracks):
  Track A (Backend):  1a → 1b → 1c → 1e     [Chatbot backend]
  Track B (Frontend): 1d                       [Chatbot frontend wiring]
  Track C (DevOps):   2a → 2b → 2c            [Cloud deployment]

Week 2 (parallel tracks):
  Track A (ML team):  3a → 3b → 3c            [MLOps/MLflow]
  Track B (DevOps):   2d                       [CI/CD deploy workflow]
```

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Chatbot not functional → lose points in 5/10 bucket | **HIGH** | Start here, even a basic LLM chat that knows user preferences scores well |
| MLflow setup issues | Medium | Fallback: manual experiment log in a notebook |
| Deployment infra cost | Low | Use free tiers (Railway/Render/Fly.io) |
