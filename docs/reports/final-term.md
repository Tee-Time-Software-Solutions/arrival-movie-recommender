# Group Final Report — Arrival Movie Recommender

---

## 1. Did the group create a recommender / chatbot system?

**Midterm recap:** We had a solid ALS architecture and an 8-step offline pipeline, but the critical gap was that `Recommender` on `main` was returning hardcoded mock data. The real ML code lived on a feature branch, not in production.

**What changed:** That gap is closed. The recommender is now fully integrated end-to-end on `main`.

---

### The offline pipeline

The backbone of the system is an ALS (Alternating Least Squares) collaborative filtering model trained on MovieLens 20M. The pipeline runs 10 sequential steps:

1. Preprocess movies (memory-efficient dtypes, genre parsing)
2. Preprocess ratings (rating → preference mapping)
3. Fetch live app swipes from Postgres into parquet
4. Merge MovieLens ratings with real app interactions
5. Filter sparse users/movies (minimum interaction thresholds)
6. Prune movie metadata to training set
7. Chronological split (train / val — respects temporal order, no data leakage)
8. Build sparse confidence matrix (`C = 1 + α·|preference|`)
9. Train ALS (`R ≈ U·Vᵀ`, solved via alternating closed-form least squares)
10. Evaluate: Precision@10, Recall@10, NDCG@10 (with seen-movie filtering)

The chronological split strategy — as opposed to random splits — ensures the pipeline reflects how the model will perform in production, where past interactions train and future ones are predicted.

Baseline metrics (ALS, factors=64):

| Metric | Score |
|---|---|
| Precision@10 | 0.055 |
| Recall@10 | 0.081 |
| NDCG@10 | 0.183 |

---

### Second ML model: FM with BPR loss

We added a second offline pipeline using Factorization Machines (FM) via LightFM with Bayesian Personalised Ranking (BPR) loss. Where ALS models absolute preferences, BPR focuses on relative ordering — "item i should rank above item j for user u" — making it particularly well-suited to implicit feedback like swipes. The FM pipeline shares the same base preprocessing steps as ALS, then diverges for data format, training, and evaluation. Both models run on the same data splits, which makes comparison meaningful.

---

### Online serving and real-time learning

Once ALS embeddings are trained and saved as `.npy` artifacts, the online system takes over:

- **User state resolution:** Redis hot-cache → Postgres persistent store → ALS-trained embedding → cold start (mean of all user embeddings). The fallback chain ensures no user ever gets a broken experience.
- **Ranking:** Fully vectorised dot-product scoring (`movie_embeddings @ user_vector`) with `argpartition` for O(N) top-k selection at scale.
- **Online feedback loop:** Every swipe updates the user vector in-place via a gradient-like rule (`η=0.05`, norm cap 10.0), persisted to Redis and asynchronously to Postgres.
- **Seen-movie filtering:** A per-user Redis set tracks seen movies and excludes them from ranking.

---

### Knowledge Graph — explainability layer

We built a Neo4j-based Knowledge Graph that enriches recommendations with semantic context. The graph models Movies, Persons (directors, actors, writers), Genres, Keywords, ProductionCompanies, and Collections as nodes, all connected by typed relationships.

On top of the graph, we built:

- **Beacon map:** A per-user entity profile derived from swipe history. Each entity (director, actor, genre, keyword) gets a weighted score based on swipes — likes push it up, dislikes push it down, with entity-type multipliers (Director: 1.5×, Actor: 1.0×, Genre: 0.8×, Keyword: 0.5×). Cached in Redis with 24h TTL.
- **Graph traversal:** Given a user's beacon map and a recommended movie, the traversal finds the highest-scoring path connecting them (e.g., user liked Nolan films → this movie is directed by Nolan).
- **Explanation renderer:** Converts the scored path into a human-readable sentence with structured entity references for the frontend.

This addresses a real limitation of pure collaborative filtering: it tells you what to watch, but not why. The Knowledge Graph closes that gap.

---

### Closed-loop data pipeline

The offline pipeline fetches real app swipes from Postgres before every training run (`fetch_app_swipes.py`). App user IDs are offset by 10,000,000 to avoid colliding with MovieLens user IDs. The merge step deduplicates interactions (keeping the latest per user–movie pair) and excludes skips from the training matrix. This means every pipeline run incorporates real user behavior, not just static MovieLens data — a true closed loop between the live product and the training pipeline.

---

## 2. Did the group do an EDA?


> **Image:** [EDA notebook — rating distribution and sparsity analysis]

---

## 3. Did the group do a deployment?

**Midterm recap:** Docker setup was real and in the repo. Cloud infra claims (Terraform + Ansible) were described but code wasn't fully in the repo.

**What changed:** The full infrastructure is now in the repo and working.

### Docker Compose stack

Both dev and production Docker Compose files are in the repo. The stack includes:

| Service | Purpose |
|---|---|
| `backend` | FastAPI app with health checks |
| `db-migration` | Alembic migrations on start |
| `redis` | User vector cache + feed queues |
| `neo4j` | Knowledge Graph database |
| `nginx` | Reverse proxy for frontend + backend |
| `prometheus` | Metrics scraping |
| `grafana` | Dashboards + alerting |
| `mlflow` | Experiment tracking server |

> **Image:** [Grafana dashboard — API latency, request volume, error rates]

> **Image:** [MLflow UI — ALS experiment runs with metric comparison]

### Multi-cloud infrastructure

Terraform modules provision the full environment on both AWS and Azure. Ansible handles provisioning (installing Docker, deploying compose, configuring services). A custom output redirection script (`infra/scripts/output_redirection/`) reads Terraform JSON outputs and injects them into Jinja2-templated `.env` files and Ansible inventory — fully automating the handoff between infrastructure and application configuration.

Azure-specific: credentials are fetched from Azure Key Vault via Managed Identity (`DefaultAzureCredential`), so no secrets are stored in environment files or the image.

> **Image:** [Project architecture — multi-cloud deployment diagram]

---

## 4. Did the group have MLOps?

**Midterm recap:** CI pipeline was solid, offline/online separation was clean. Missing: experiment tracking, model registry, monitoring.

**What changed:** All three gaps were addressed.

### Experiment tracking — MLflow

Every ALS training run is tracked in MLflow:

- Hyperparameters logged: `factors`, `regularization`, `iterations`, `alpha`, `min_user_ratings`
- Metrics logged: `recall@10`, `precision@10`, `ndcg@10`, `pipeline_duration_min`
- Runs are grouped under the `ALS_Recommender_Offline` experiment

This makes every training run reproducible and comparable. Miguel can browse the experiment history and see exactly which config produced which metrics.

> **Image:** [MLflow experiment tracking — run comparison across configs]

### Training notifications — Discord

Every pipeline run sends a structured report to Discord: metrics summary, hyperparameters, and an auto-generated bar chart of the evaluation scores. This is triggered from `RecommenderPipeline._notify()` — a base class method inherited by both ALS and FM pipelines.

> **Image:** [Discord notification — ALS training run report with chart]

### Monitoring — Grafana + Prometheus

Grafana is provisioned with dashboards and alerting rules out of the box (provisioning files in `deployment/telemetry/grafana/`). Prometheus scrapes metrics from the FastAPI backend. The stack spins up alongside the application in the same Docker Compose.

### CI pipeline

GitHub Actions runs on every push:

1. `lint` job: installs dependencies via `uv`, runs `ruff` format check
2. `unit-tests` job: runs after lint, executes the full test suite

Pre-commit hooks run on every commit: trailing whitespace, YAML/JSON validation, large file detection, debug statement detection, `gitleaks` secret scanning, and unit tests.

### Offline pipeline scheduling

The ALS pipeline has a built-in APScheduler entry point (`run_pipeline_cron_job`) with a Redis distributed lock to prevent concurrent runs across workers. This means the pipeline can be scheduled on any worker without risk of two instances running simultaneously.

---

## 5. Are the group following good development practices?

**Midterm recap:** Full marks. Trunk-based development, CODEOWNERS, Gemini bot + human reviews, 43 issues, pre-commit hooks with gitleaks. Genuinely professional for a student project.

**What changed:** Maintained and extended. As of the final submission:

- 80+ PRs merged, all with Gemini bot reviews and human sign-off on critical ones
- CODEOWNERS still in place — PRs to backend auto-request `javidsegura`, frontend `aallendez`, recommender the data team
- Pre-commit hooks extended (unit tests added to the hook chain)
- Feature branches for every piece of work: `feat/knowledge_graph_v1`, `closing-loop`, `new-offline-model`, `EDA`, `feat/azure`, `feat/deployment` — all merged via PR
- Gemini review feedback acted on in follow-up commits (`fix: gemini feedback`, `fix: gemini feedback2`) — not just dismissed
- `uv` throughout for reproducible environments

> **Image:** [CI/CD pipeline — GitHub Actions passing on a PR]

---

## 6. Evidence of exceptional group ability

> *(Evaluated by Miguel — not described here)*

What we'd point to: a full closed-loop ML system (swipe → Postgres → offline retrain → new embeddings → serving), a Knowledge Graph for recommendation explainability, two ML algorithms trained on the same pipeline with shared base steps, MLflow + Grafana + Discord for complete observability across the ML lifecycle, and multi-cloud deployment (AWS + Azure) with automated config injection. The tooling choices throughout — `uv`, `ruff`, `gitleaks`, Grafana provisioning-as-code, Azure Managed Identity — reflect production engineering judgment, not just getting something to work.
