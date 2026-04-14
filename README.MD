# Arrival — Movie Recommender

> A production-grade movie recommendation platform with a Tinder-style swipe interface, online learning, Knowledge Graph explainability, and a full MLOps stack.

![CI](https://github.com/Tee-Time-Software-Solutions/arrival-movie-recommender/actions/workflows/ci-backend.yml/badge.svg)

---

## What it is

Users swipe through movies (like, dislike, supercharge, skip). Every swipe updates their personal embedding in real time. The recommender — an Alternating Least Squares (ALS) collaborative filtering model trained on MovieLens 20M + live app interactions — ranks the next batch accordingly.

A Neo4j Knowledge Graph enriches each recommendation with an explanation: "You might like this because you enjoyed films directed by Christopher Nolan."

> **Image: Architecture overview**
> `[architecture.png]`

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Python 3.11, SQLAlchemy Core, asyncpg |
| Database | PostgreSQL 16 |
| Cache / queues | Redis 7.2 |
| Knowledge Graph | Neo4j 5 |
| Frontend | React 19, TypeScript, Vite, Zustand, Framer Motion, Tailwind CSS v4 |
| Auth | Firebase (Google OAuth + email/password) |
| ML — offline | ALS via `implicit`, FM via LightFM (BPR) |
| ML — online | Gradient-like user vector updates on every swipe |
| Experiment tracking | MLflow |
| Monitoring | Grafana + Prometheus |
| Notifications | Discord webhook |
| Containerisation | Docker Compose (dev + production) |
| Infrastructure | Terraform (AWS + Azure), Ansible |
| Secrets | Azure Key Vault (Managed Identity), AWS Secrets Manager |
| CI | GitHub Actions (lint → unit tests) |
| Package manager | `uv` |

---

## Repository structure

```
├── backend/
│   └── src/movie_recommender/
│       ├── api/v1/                  # FastAPI routers (one per domain)
│       ├── core/                    # Settings, logger, clients
│       ├── database/
│       │   ├── models.py            # SQLAlchemy Core table definitions
│       │   ├── CRUD/                # Async data access (one file per entity)
│       │   └── migrations/          # Alembic
│       ├── dependencies/            # FastAPI Depends() factories
│       ├── schemas/                 # Pydantic I/O + TypedDicts for DB rows
│       └── services/
│           ├── recommender/
│           │   └── pipeline/
│           │       ├── offline/     # ALS + FM training pipelines
│           │       │   ├── models/als/   # ALS steps + MLflow integration
│           │       │   └── models/fm/    # LightFM (BPR) steps
│           │       └── online/      # Serving: ranker, user_state, feedback
│           ├── knowledge_graph/     # Neo4j schema, beacon, traversal, renderer
│           ├── infra/               # Cloud credential helpers (Azure, AWS)
│           └── notifiers/           # Discord training reports
├── frontend/app/src/
│   ├── app/                         # Page components
│   ├── components/                  # Feature + UI components
│   ├── stores/                      # Zustand stores (auth, movie, chat)
│   └── services/api/               # Axios API layer
├── deployment/
│   ├── docker-compose.yml           # Base compose (shared services)
│   ├── docker-compose.dev.yml       # Dev overrides
│   ├── docker-compose.remote.yml    # Production overrides
│   └── telemetry/
│       ├── prometheus.yml
│       └── grafana/provisioning/    # Dashboards, datasources, alerting (as code)
├── infra/
│   ├── terraform/
│   │   ├── aws/                     # VPC, EC2, RDS, S3
│   │   └── azure/                   # VNet, VM, MySQL, Key Vault
│   ├── ansible/                     # Provisioning playbooks
│   └── scripts/
│       ├── output_redirection/      # Terraform outputs → .env + Ansible inventory
│       └── resource_connections/    # Cloud connection helpers
└── makefile                         # Dev commands
```

---

## ML pipeline

### Offline training

Two models share the same base preprocessing steps:

```
Preprocess movies → Preprocess ratings → Fetch app swipes (Postgres)
  → Merge interactions → Filter → Prune → Chronological split
    → [ALS] Build sparse matrix → Train ALS → Evaluate → MLflow + Discord
    → [FM]  Build LightFM data → Train BPR → Evaluate
```

The chronological split ensures no data leakage between train and validation — a critical distinction for time-ordered interaction data.

Every ALS run is tracked in MLflow (hyperparameters + metrics + pipeline duration) and a Discord notification is sent on completion.

> **Image: MLflow experiment tracking**
> `[mlflow.png]`

> **Image: Discord training notification**
> `[discord.png]`

Evaluation metrics (ALS, factors=64, MovieLens 20M):

| Metric | Score |
|---|---|
| Precision@10 | 0.055 |
| Recall@10 | 0.081 |
| NDCG@10 | 0.183 |

### Online serving

Every swipe triggers:

1. Load user vector: Redis → Postgres → ALS embedding → cold start (mean of all embeddings)
2. Apply feedback update (`η=0.05`, norm cap 10.0)
3. Persist: Redis (immediate) + Postgres (async background task)
4. On next recommendation request: vectorised dot-product ranking with `argpartition` (O(N) top-k) and seen-movie exclusion

### Closed loop

The offline pipeline fetches live swipes from Postgres before every training run. App user IDs are offset by 10,000,000 to avoid collision with MovieLens IDs. Every retrain incorporates real user behavior alongside the base MovieLens 20M corpus.

### Knowledge Graph

Neo4j graph: Movies, Persons (director/actor/writer), Genres, Keywords, Collections, ProductionCompanies.

- **Beacon map** — per-user entity weighting derived from swipe history (entity-type multipliers, Redis-cached 24h)
- **Traversal** — finds highest-scoring path from user interests to recommended movie
- **Renderer** — converts scored path to a human-readable explanation sentence

---

## Monitoring

> **Image: Grafana dashboard**
> `[grafana.png]`

Grafana dashboards and alerting rules are provisioned as code (`deployment/telemetry/grafana/provisioning/`). Prometheus scrapes FastAPI metrics. The full observability stack starts with the application via Docker Compose.

---

## Setup

### Prerequisites

- Docker + Docker Compose
- Python 3.11 (managed by `uv`)
- Node 20+
- Firebase project (for auth)

### 1. Environment variables

```bash
# Backend
cp backend/env_config/base/.env.dev backend/env_config/synced/.env.dev
# Fill in: POSTGRES_URL, REDIS_URL, NEO4J_URI, NEO4J_PASSWORD,
#          FIREBASE_PROJECT_ID, TMDB_API_KEY, DISCORD_WEBHOOK_PIPELINE

# Frontend
cp frontend/app/.env.example frontend/app/.env.local
# Fill in: VITE_BASE_URL, VITE_FIREBASE_API_KEY, ...
```

### 2. Install dependencies

```bash
make install
```

### 3. Start dev environment

```bash
make dev-start
```

| Service | URL |
|---|---|
| App | http://localhost |
| API | http://localhost/api/v1 |
| API docs | http://localhost:8000/docs |
| Grafana | http://localhost:3000 |
| MLflow | http://localhost:5000 |
| Neo4j browser | http://localhost:7474 |

### 4. Run the offline pipeline

```bash
# ALS (primary model, MLflow-tracked)
cd backend
python -m movie_recommender.services.recommender.pipeline.offline.models.als.main

# FM / LightFM BPR (requires: pip install lightfm)
python -m movie_recommender.services.recommender.pipeline.offline.models.fm.main
```

Trained embeddings are saved to `model_assets_dir` (configured in `config.yaml`). The online serving layer loads them at startup.

---

## Development commands

```bash
make install          # Install all dependencies (backend + frontend)
make dev-start        # Start dev environment (Docker Compose)
make dev-rebuild      # Rebuild images and start
make dev-stop         # Stop dev environment
make backend-tests    # Run unit tests
make format-check     # Lint check (ruff)
make gen-dev-token    # Generate Firebase token for local testing
```

### Database migrations

```bash
cd backend
alembic revision --autogenerate -m "description"
alembic upgrade head
```

---

## CI/CD

> **Image: CI/CD pipeline**
> `[ci-cd.png]`

GitHub Actions runs on every push touching `backend/`:

1. **Lint** — `ruff` format check via `uv`
2. **Unit tests** — full test suite, runs after lint passes

Pre-commit hooks (run on every local commit):

- Trailing whitespace, YAML/JSON validation, large file detection
- Debug statement detection
- `gitleaks` secret scanning
- Unit tests

CODEOWNERS assigns reviewers automatically: backend → `@javidsegura`, frontend → `@aallendez`, recommender → data team.

---

## Infrastructure

Multi-cloud deployment (AWS + Azure) via Terraform + Ansible:

```
Terraform (AWS or Azure)
  → Provisions VPC/VNet, compute, managed DB, secrets manager
  → terraform output -json
      → infra/scripts/output_redirection/
          → .env files (backend + frontend)
          → Ansible inventory
Ansible
  → Installs Docker on EC2 / Azure VM
  → Deploys production Docker Compose
  → Configures all services
```

Azure deployments use Managed Identity + Key Vault: no credentials in environment files or Docker images. AWS deployments use Secrets Manager.

---

## Team

| Name | Role |
|---|---|
| Javier Dominguez Segura | Backend / ML Engineer / MLOps |
| Lucas Van Zyl | Data Engineer |
| [Teammate] | Data Scientist |
| [Teammate] | ML Engineer |
| [Teammate] | Frontend |
| [Teammate] | Frontend |

---

## References

- [TMDB API](https://www.themoviedb.org/documentation/api)
- [MovieLens 20M dataset](https://grouplens.org/datasets/movielens/20m/)
- [implicit (ALS)](https://github.com/benfred/implicit)
- [LightFM (BPR/FM)](https://github.com/lyst/lightfm)
- [MLflow](https://mlflow.org)
