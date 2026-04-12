# ROADMAP

**Due: 2026-04-14 EOD | Budget: ~25h | Presentations: 2026-04-15 & 2026-04-17**

---

## Current State (as of 2026-04-11)

### What's working on main
- Offline pipeline: ALS + FM (LightFM) both implemented
- Serving layer: artifact_loader → ranker (dot-product) → online updater (gradient step)
- Unit tests with synthetic fixtures — run without any backend services
- Knowledge graph (Neo4j): schema, writer, traversal, explainer
- EDA notebook: merged via PR #75
- Docker Compose: dev + prod
- CI: lint + unit tests on every push

### Key gap (from prof's feedback)
- Recommender artifacts must be on main and actually wired to the API
- Missing: chatbot, third model, CD pipeline
- Grade risk: "exceptional ability" = 0/1 — needs end-to-end working system

---

## Phase 1 — CD Pipeline [ ~4h ]

**Goal:** GitHub Actions workflow that builds + deploys on merge to main.

### Tasks
- [ ] `1.1` Write `.github/workflows/cd-backend.yml`: build Docker image → push to registry → deploy
- [ ] `1.2` Choose target: AWS ECS (Fargate) or Azure Container Apps (whichever is already set up)
- [ ] `1.3` Add env secrets to GitHub repo (registry creds, deploy keys)
- [ ] `1.4` Ensure `docker-compose.yml` reflects the same config as the deployed container

### Design notes
- Trigger: `push` to `main` with `paths: backend/**` or `deployment/**`
- Requires: Docker registry (ECR or ACR) + running cluster
- Keep it minimal: build → push → trigger re-deploy (rolling update)

---

## Phase 2 — Third Model + Benchmarking [ ~6h ]

**Goal:** 3 ranked models, one comparison table, "at least 3 models" fulfilled.

### Models
| Model | Status | Library |
|---|---|---|
| ALS (Implicit) | done | `implicit` |
| FM (LightFM) | done | `lightfm` |
| Popularity Baseline | to do | `pandas` only |

### Why Popularity Baseline?
- Zero ML complexity → implements fast (~1h)
- Essential lower-bound in any proper benchmarking setup
- Explicitly teaches the concept of baselines (slide corpus concept: always beat a naive baseline)
- Easy to extend: time-decay popularity, genre-conditional popularity

### Tasks
- [ ] `2.1` Implement `learning/offline_pipelines/popularity.py`
  - Score = count of interactions in train set (optionally time-decayed)
  - Output: ranked list of top-N movie IDs per genre / global
- [ ] `2.2` Add popularity to `compare_als_vs_fm.py` → rename to `compare_models.py`
- [ ] `2.3` Add Precision@10, Recall@10, NDCG@10 for all 3 models side by side
- [ ] `2.4` Write `tests/unit/test_popularity.py`

### Recommender math cheatsheet
**ALS:**
- Factorizes U (users) × M (movies) → embeddings U, V ∈ Rᵈ
- Implicit confidence: c(u,i) = 1 + α·r(u,i), preference p(u,i) = 1 if r>0 else 0
- Loss: Σ c(u,i)(p(u,i) − uᵤᵀvᵢ)² + λ(||U||² + ||V||²)
- Online update: uᵤ ← uᵤ + η·pref·vᵢ, capped at norm_cap
- Serving: score(u,i) = uᵤᵀvᵢ (dot product)

**FM (LightFM):**
- ŷ(x) = w₀ + Σᵢwᵢxᵢ + Σᵢ<j ⟨vᵢ,vⱼ⟩xᵢxⱼ
- x = one-hot(user) ⊕ one-hot(item) ⊕ side_features (genres, tags)
- WARP loss: optimizes ranking directly (sample negative until it violates margin)
- Advantage over ALS: can use item side features → better cold start

**Popularity Baseline:**
- score(i) = |{u : r(u,i) > 0 in train}|
- No personalization; useful as a floor to beat

**Metrics (Precision, Recall, NDCG at K=10):**
- Precision@K = |recommended ∩ relevant| / K
- Recall@K = |recommended ∩ relevant| / |relevant|
- NDCG@K = DCG@K / IDCG@K, DCG = Σ rel_k / log₂(k+1)

---

## Phase 3 — Backend Functionality [ ~4h ]

**Goal:** Add the backend features that are currently missing or incomplete.

### Tasks
- [ ] `3.1` **Rate limiter** — FastAPI middleware using `slowapi` (token-bucket per user/IP)
- [ ] `3.2` **Full-text search** — `GET /api/v1/movies/search?q=` using PostgreSQL `tsvector` + GIN index
- [ ] `3.3` **DB indexing** — add missing indexes (movies.title, movies.tmdb_id already has unique, check swipes)
- [ ] `3.4` **Chatbot endpoint** — `POST /api/v1/chat` wired to the LLM agent (see Phase 4)
- [ ] `3.5` **Recommendations explanation endpoint** — `GET /api/v1/people/top` already implemented via KG; confirm it's exposed

### Design notes
- Rate limiter: 60 req/min for /feed, 10 req/min for /chat, 200 req/min for everything else
- Full-text search: `CREATE INDEX ON movies USING GIN(to_tsvector('english', title))`, new migration
- Don't add Redis caching layer on top of what already exists — feed queue IS the cache

---

## Phase 4 — Chatbot Integration [ ~8h ]

**Goal:** LLM-based chatbot that uses the recommender as a tool. Minimal but functional.

### Architecture
```
User message
    ↓
LLM agent (openrouter: free open-source model e.g. mistral-7b or llama-3)
    ↓ tool call
Recommender.get_top_n_recommendations(user_id, candidate_ids)
    ↓
LLM formats & explains the recommendations
    ↓
Response to user
```

### Tech stack
- **LLM**: OpenRouter (free tier) — `mistral-7b-instruct` or `llama-3-8b-instruct`
- **Agent framework**: LangGraph (simplest stateful agent) or plain function-calling
- **Memory**: in-process dict (same as Recommender.online_user_vectors) — no external DB needed
- **MCP**: skip for now (overcomplicated for 25h budget)

### Tasks
- [ ] `3.1` Create `services/chatbot/main.py` — agent loop with tool call to recommender
- [ ] `3.2` Wire chatbot endpoint: `POST /api/v1/chat` (message → response)
- [ ] `3.3` Write `tests/unit/test_chatbot.py` with mocked LLM
- [ ] `3.4` Document: what role does the chatbot play (enhanced search / recommendation explainer)

---

## Phase 4 — Polish for Grade [ ~3h ]

- [ ] `4.1` Ensure recommender artifacts are committed or auto-generated via Makefile target
- [ ] `4.2` Verify end-to-end: swipe → feedback → recommendation updates
- [ ] `4.3` Update group report to reflect what's on main (not branches)
- [ ] `4.4` EDA notebook: confirm it renders and has the plots the prof expects

---

## Testing the pipeline without the backend

Run unit tests (no Docker, no Postgres, no Redis, no Firebase needed):

```bash
cd backend
uv run pytest tests/unit/ -v
```

These use synthetic fixtures from `tests/conftest.py` — small deterministic embeddings, no disk I/O.

To run the full offline pipeline (needs MovieLens data on disk):

```bash
cd backend
uv run python -m movie_recommender.services.recommender.learning.offline_pipelines.implicit_als
# or
uv run python -m movie_recommender.services.recommender.learning.offline_pipelines.factorization_machines
```

---

## Benchmarking question

Yes — benchmarking is expected and it already partially exists in `compare_als_vs_fm.py`.
The goal is: one table with Precision@10, Recall@10, NDCG@10 for all 3 models on the same val set.
The popularity baseline is the key missing piece.

---

## Do I have to personally add more models?

You have ALS + FM = 2. The CLAUDE.md says "at least 3 different models". Add popularity baseline as the third — it's:
- Defensible academically (standard baseline)
- Fast to implement
- Required to claim benchmarking was done properly
