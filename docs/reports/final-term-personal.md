# Personal Final Report — Javier Dominguez Segura

---

## 1. Individual contributions

**Midterm recap:** Led the entire backend — FastAPI, TMDB hydration, Firebase auth, SQLAlchemy/Alembic, Redis feed queue, pre-commit hooks, CI, stateful backend. Scored 2/2.

**What changed:** This term I moved into ML and infrastructure — the areas the professor specifically called out.

### What I built for the final

**Recommender integration (the critical fix)**
The core gap from midterm was that `Recommender` on `main` returned mock data. I closed it. The full user state resolution chain (Redis → Postgres → ALS embedding → cold start), the vectorised ranker with `argpartition`, the online feedback updater — all mine.

**Online learning pipeline**
- `feedback.py`: maps swipe actions to preference scores (supercharged doubles the signal)
- `updater.py`: gradient-like user vector update rule
- `ranker.py`: vectorised dot-product scoring with O(N) top-k via `argpartition`
- `user_state.py`: base embedding + cold start fallback logic

**Closed-loop data pipeline**
- `fetch_app_swipes.py`: exports live app interactions from Postgres to parquet before every offline training run (with APP_USER_ID_OFFSET=10,000,000 to avoid collision with MovieLens user IDs)
- `merge_interactions.py`: deduplicates interactions across MovieLens + app data, excludes skips from training

**Knowledge Graph service**
- `schema.py`: Neo4j uniqueness constraints for all node types (idempotent, with transient-error retry and jitter to handle concurrent worker startup)
- `beacon.py`: per-user entity weighting from swipe history (entity-type multipliers, cached in Redis)
- `traversal.py`: graph path scoring and selection
- `renderer.py`: converts scored paths to structured explanation text
- `explainer.py`: orchestrates beacon → traversal → render

**MLflow integration**
Wired MLflow into the ALS pipeline: logs all hyperparameters, evaluation metrics, and pipeline duration. Every run is grouped under `ALS_Recommender_Offline` experiment.

**Discord notifier**
`DiscordNotifier.send_training_report()`: sends structured training summaries (metrics table + bar chart) to Discord via webhook. Auto-generates a matplotlib bar chart, uploads it as a file attachment, then cleans up the temp file.

**Infrastructure**
- `services/infra/azure.py`: Azure Key Vault secret fetching via Managed Identity (`DefaultAzureCredential`) — no credentials in env
- `infra/scripts/output_redirection/`: Terraform output → Jinja2 template injection → `.env` files + Ansible inventory
- Azure + AWS Terraform working (PR: azure and aws working)

**Base pipeline steps (shared by ALS + FM)**
`preprocess_movies`, `preprocess_ratings`, `filter`, `prune_movies`, `split`, `merge_interactions`, `fetch_app_swipes` — all base steps that both ALS and FM pipelines import.

---

## 2. What I did on top of the work of others

**Midterm recap:** API endpoint specs + docstrings let the frontend build without waiting on me. Docker Compose was the integration point for all teams. Scored 2/2.

**What changed:** The pattern continued but in a different direction — now connecting the ML team's offline artifacts to the live serving layer.

The recommender integration was exactly this. The ML team trained ALS and produced embeddings. I built everything that takes those `.npy` files and makes them available in production: the artifact loader, the serving layer, the Redis caching strategy, the Postgres persistence. My work is the bridge between "model trained" and "model serving."

The Knowledge Graph also extends everyone's work. The data team built preprocessing that extracts movie metadata. I took that metadata and built a semantic layer on top of it — now recommendations have explanations tied to actors, directors, genres, and keywords from the movie data they processed.

The Discord notifier makes the ML team's training runs visible to everyone in the group without anyone having to check logs. Every training run lands in a shared channel.

The output redirection script takes the infrastructure team's Terraform outputs and automatically configures both the backend `.env` files and the Ansible inventory. That removed a manual step that was causing deployment inconsistencies.

---

## 3. What I did to help others

**Midterm recap:** API specs, Figma for frontend, mentoring teammates who were falling behind, design suggestions for chatbot role. Scored 1.5/2 — feedback was to be "more explicit about expectations."

> **This section needs your input.** Based on the midterm feedback, the professor wanted more explicit examples of how you set expectations and helped specific teammates. I'd suggest covering:
>
> - Did you pair-program or review PRs for specific teammates? Name them + what the PR was.
> - Did you write detailed PR comments / Gemini follow-up explanations to help the data team understand why changes were needed?
> - Did you define data contracts (API shapes, parquet schemas) that unblocked multiple people at once?
> - Did you resolve merge conflicts or help anyone unstuck on a specific technical problem?
> - Did you act on the midterm feedback about being clearer with expectations? How? (Written expectations, team agreements, GitHub issues with clear acceptance criteria?)
>
> A few concrete examples here go a long way. The professor already knows you're the backbone — he wants to see you demonstrably lifted the people around you.

---

## 4. What I could have done better

**Midterm recap:** "Spent too much time on backend/frontend and not enough on the recommender core." The professor specifically said to get into ML and chatbot for the final. Scored 1.5/2.

**What I actually improved:**
I delivered on the ML side. The recommender is now real, the online learning is working, I built the Knowledge Graph, MLflow is in, Discord notifications are working. That's the direct response to the midterm criticism.

**What I still didn't get to:**
The chatbot. It's in the CLAUDE.md as a core item — open-source model, memory, MCP integration — and I didn't build it. In hindsight, I underestimated how much the recommender integration would take to do properly, and the Knowledge Graph added significant scope. Given another sprint I would have started the chatbot sooner and in parallel, possibly delegating the KG renderer to someone else.

**What I'd do differently architecturally:**
The pipeline refactor (`80eb0362 refactor: initial commit`) caused the EDA notebook to get deleted and needed to be re-added on a new branch. That's a process issue — when doing a major refactor, preserve working artifacts explicitly rather than letting them get swept. A simple list of "files to keep" in the PR description would have caught it.

**On team management:**
The midterm feedback about being clearer with expectations was fair. I gave people direction but often implicitly — I assumed they'd infer what "done" looked like. For the final I tried to be more explicit with acceptance criteria in issues, but I probably still relied too much on Slack/verbal alignment over written agreements that survive context switches.

The honest version: I could deliver technically on anything the project needed, but translating that into structured expectations for others — especially teammates with less experience — is a skill I'm still developing.
