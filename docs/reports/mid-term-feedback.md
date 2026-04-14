### Group Grade: 7.5/10



**Did they create a recommender/chatbot? (4/5)**



The team built a collaborative filtering recommender using Implicit ALS trained on MovieLens 20M. The architecture is well-designed: an offline pipeline (preprocess, filter, chronological split, train, evaluate) producing 64-dim user/movie embeddings, and an online pipeline with real-time user vector updates via gradient-like rule (eta=0.05, norm cap 10.0). The FIFO queue system (FeedManager + Redis) for decoupling recommendation generation from serving is sophisticated. TMDB hydration enriches movie metadata. A Tinder-like swipe interface drives user feedback.



However, a critical finding: **on main, the Recommender class still returns mock data** (`return [(1, "Arrival"), (2, "Interstellar"), (3, "The Matrix")]` in https://github.com/Tee-Time-Software-Solutions/arrival-movie-recommender/blob/887c4864ebd6f805e9ee1df2f02649a23b6b8d56/backend/src/movie_recommender/services/recommender/main.py#L22). The real ML integration exists on the `ml-pipeline-testing-v2` branch (PR #31, still open) but was merged and then reverted (PR #29). So the live system on main does not actually use the trained model. The offline pipeline code is merged (PR #16) but the serving layer connecting it to the API is not. This is a significant gap between what the report describes and what main delivers.



The offline pipeline itself (in `learning/offline_pipeline.py`) is clean and well-structured: 8-step pipeline from preprocessing through evaluation. The evaluation code properly computes Precision@10, Recall@10, and NDCG@10 with seen-movie filtering. Reported baseline: Precision@10=0.055, Recall@10=0.081, NDCG@10=0.183. These are honest, reasonable numbers for ALS on ML-20M.



Score: 4/5. Strong architecture and real ML work, but the integration is not on main. Partial credit because the code exists and is verifiable on branches.



**Did they do data prep & EDA? (0.5/1)**



The report describes EDA on MovieLens 20M (rating distributions, sparsity analysis, user/movie activity distributions). However, there is **no EDA notebook in the repo** (no .ipynb files found anywhere). The `data/exploratorydataanalysis` branch exists but contains no EDA files, just an older snapshot of the codebase. The preprocessing scripts (`preprocess_movies.py`, `preprocess_ratings.py`, `filtering.py`) are solid data engineering work with proper rating-to-preference mapping and memory-efficient dtypes. But EDA without evidence in the repo gets only partial credit.



Score: 0.5/1. Good preprocessing, but no EDA notebook or analysis artifacts in the repo.



**Did they do a deployment? (1/1)**



Docker Compose files exist for both dev and production (`deployment/docker-compose.yml`, `docker-compose.dev.yml`). The report mentions multi-cloud (AWS + Azure) with Terraform and Ansible. While the Terraform/Ansible code is not in this repo, the Docker setup with Redis, backend, migrations, and health checks is present. Dockerfiles exist for both frontend and backend. The report's claim of staging + production environments is plausible given the infrastructure described.



Score: 1/1. Docker-based deployment is real and in the repo. Cloud infra claims are reasonable.



**Did they have MLOps? (1/1)**



The offline pipeline is reproducible (`offline_pipeline.py`), with clear artifact outputs (user/movie embeddings as .npy, mappings as .json). The separation of offline training from online serving is a proper MLOps pattern. The CI pipeline runs backend tests on every push (verified: `ci-backend.yml` with lint + unit-test jobs, using uv). Pre-commit hooks include formatting, secret scanning (gitleaks), and unit tests. The Gemini PR reviewer integration is a nice touch. Missing: no experiment tracking (MLflow/W&B), no model registry, no data versioning.



Score: 1/1. Solid for a mid-term. The offline/online separation and CI pipeline are well-executed.



**Good development practices? (1/1)**



Excellent. Trunk-based development with feature branches (e.g., `feat/stateful_backend`, `feat/firebase_auth`, `recommender-setup`). CODEOWNERS file assigns reviewers automatically (backend -> javidsegura, frontend -> aallendez, recommender -> LAIN-21/SnileMB/Diechi09). PRs with reviews: 10 merged PRs, Gemini bot reviews on all, human reviews on key ones. Pre-commit hooks with formatting, linting, secret scanning. CI runs on every push. 43 GitHub issues created. Proper `.gitignore`, env config examples. Using `uv` as package manager with Makefiles. This is genuinely professional-grade tooling for a student project.



Score: 1/1.



**Evidence of exceptional ability as a group? (0/1)**



The architecture is ambitious and well-designed, but the gap between the report and main (mock data still in production code) is concerning. The group report reads as if the system is fully integrated, but it is not. The CI/CD and tooling are excellent, but the core product (recommendations) does not work on main. For exceptional, I would need the ML pipeline fully integrated and working end-to-end.



Score: 0/1.



#### Javier Dominguez Segura: 8.5/10



Javier (`javidsegura`) is the team's top contributor with 29 contributions and 22 commits. He built the entire backend: FastAPI, TMDB hydration, Firebase auth, SQLAlchemy/Alembic migrations, CRUD operations, the FeedManager + Redis queue system, pre-commit hooks, CI, and the stateful backend with PostgreSQL (PR #33, 4450 additions). He also defined the Recommender class API and created endpoint specs for the frontend team. The appendix confirms him as #1 contributor (13,097++, 4,635--).



Your code quality is good. Async patterns, dependency injection, structured settings, Redis, Firebase auth with token verification. This reads like professional code, not a student project. I really like how you unblocked the whole team by writing API specs with detailed docstrings so others could build against your interfaces without waiting. That is how senior engineers think. Also, great job on the leadership side. You mentored teammates who were falling behind, with empathy and a focus on helping them move forward rather than pointing fingers. That mix of strong technical skills and good people skills is not common.



You said it yourself: you spent too much time on backend/frontend and not enough on the recommender. For the final, get into the ML and chatbot side. You clearly have the skills to make the integration work. Also, your teammates mentioned you could be more explicit about what you expect from them. That is worth taking seriously. Good tech leads set clear, written expectations early so nobody gets surprised later.



**Individual contributions (2/2):** See above.



**On top of others (2/2):** The backend sits at the intersection of frontend and recommender. He created API endpoint specs in a Google doc for the frontend team, then implemented the function signatures with detailed docstrings so others could understand and keep working. He built on top of the Docker Compose that was touched by all teams. This is real integration work that unblocked the whole team.



**Help others (1.5/2):** Suggested Figma for frontend collaboration. Defined the API interface between backend and service layer. Gave design suggestions for the feedback system and chatbot role ("enhanced search tool"). Suggested EDA plots for the data team (e.g., frequency analysis for user preference filters). Took the lead on giving feedback and mentoring teammates when things were falling behind, with an approach focused on helping rather than blaming. The leadership and mentoring claims are consistent with him being the backbone of the project.



**Self-reflection (1.5/2):** Honest and specific: acknowledges spending too much time on backend/frontend and not enough on the recommender core. Plans to focus on chatbot integration after backend stabilizes. Also mentions receiving feedback from peers about being clearer with expectations. Good self-reflection with concrete plans. Could have been a bit more critical about specific technical decisions.



**Exceptional ability (1.5/2):** Probably the strongest individual technical contribution in the group. The backend architecture (FastAPI, SQLAlchemy/Alembic, Redis queue, Firebase auth, TMDB hydration, CODEOWNERS, pre-commit hooks with gitleaks) is professional-grade. The report is well-written, direct, and honest. Strong mix of technical leadership and team management. For the final, I want to see him contribute to the ML/chatbot side as he plans.



OVERALL GRADE: 8/10 (50% group 7.5 + 50% personal 8.5)
