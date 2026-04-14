# Recommender — Architecture

## Offline pipelines

### ALS

Run once to produce embeddings. Re-runs nightly via cron job.

```
pipeline/artifacts/dataset/source/small/
  movies.csv
  ratings.csv
      │
      ▼
fetch_app_swipes()                          [base/steps/fetch_app_swipes.py]
  query Postgres for all swipes
  map action_type + is_supercharged → preference score {-2,-1,0,+1,+2}
  shift app user_ids by APP_USER_ID_OFFSET (default 10_000_000)
    — prevents collision with MovieLens user IDs (max ~162k)
  → pipeline/artifacts/dataset/raw/swipes_from_db.parquet
  skip: SKIP_DB_SWIPE_EXPORT=1
      │
      ▼
preprocess_movies()                         [base/steps/preprocess_movies.py]
  extract year, clean title, split genres
  → dataset/processed/movies_clean.parquet

preprocess_ratings()                        [base/steps/preprocess_ratings.py]
  rating (0.5–5.0) → bucket {1,2,3,4} → preference {-2,-1,+1,+2}
  → dataset/processed/ratings_clean.parquet
      │
      ▼
merge_interactions()                        [base/steps/merge_interactions.py]
  concat MovieLens ratings + app swipes
  dedupe: keep latest timestamp per (user_id, movie_id)
  drop skips (preference == 0) — no directional signal for ALS
  → dataset/processed/ratings_unified.parquet
      │
      ▼
filter()                                    [base/steps/filter.py]
  iterative core filter until stable:
    remove movies  < 20 interactions
    remove users   < 10 interactions
  → dataset/processed/ratings_filtered.parquet

prune_movies()                              [base/steps/prune_movies.py]
  keep only movies that survived filtering
  → dataset/processed/movies_filtered.parquet
      │
      ▼
split()                                     [base/steps/split.py]
  per-user chronological split (no leakage)
  80% train / 10% val / 10% test
  → dataset/splits/train.parquet  val.parquet  test.parquet
      │
      ▼
build_sparse_matrix()                       [als/steps/matrix.py]
  CSR matrix R_train (users × movies), values = preference score
  + contiguous ID mappings
  → model_assets/R_train.npz
  → model_assets/mappings.json
      │
      ▼
train_als()                                 [als/steps/train_als.py]
  confidence: C = 1 + α·|preference|, α=15
  implicit.ALS(factors=16, reg=0.1, iters=15)
  → model_assets/user_embeddings.npy
  → model_assets/movie_embeddings.npy
  → model_assets/model_info.json
      │
      ▼
evaluate()                                  [als/steps/metrics.py]
  for each val user:
    scores = movie_embeddings @ user_vector
    mask seen movies → take top 10
    compute Precision@10, Recall@10, NDCG@10
  → model_assets/als_metrics.json
```

### Item-CF (offline-only)

Item-CF reuses the same base preprocessing/filter/split steps as ALS, then switches to
an item-similarity training/evaluation tail.

```
split()                                     [base/steps/split.py]
      │
      ▼
build_item_cf_matrix()                      [item_cf/steps/matrix.py]
  CSR matrix from train.parquet (user_id, movie_id, preference)
  + ID mappings with index_to_movie_id
  → model_assets/item_cf_train_matrix.npz
  → model_assets/item_cf_mappings.json
      │
      ▼
train_item_cf()                             [item_cf/steps/train_item_cf.py]
  cosine item-item similarity from train matrix columns
  optional positive-only interactions for similarity construction
  optional co-rater threshold + shrinkage regularization for stability
  top-K neighbor pruning per item for compact artifacts
  → model_assets/item_cf_similarity.npz
  → model_assets/item_cf_model_info.json
      │
      ▼
evaluate_item_cf()                          [item_cf/steps/metrics.py]
  rank candidates per val user (excluding train-seen items)
  relevant validation items use preference > relevance_preference_threshold
  compute Precision@10, Recall@10, NDCG@10
  skip users with no valid candidates/ground truth
  → model_assets/item_cf_metrics.json
```

### Nightly rerun

APScheduler fires `run_pipeline_cron_job()` at midnight.
Redis lock (`ml_pipeline_lock`, TTL 1h) prevents concurrent runs across workers.
The lock is always released in `finally`.

---

## Online pipeline (per request)

### Swipe → update user vector

```
POST /api/v1/interactions/{movie_id}/swipe
      │
      ▼
swipe_to_preference(action, is_supercharged)
  DISLIKE → -1 (or -2 if supercharged)
  LIKE    → +1 (or +2 if supercharged)
  SKIP    →  0 (no update)
      │
      ▼
get_user_vector(user_id)  — three-tier lookup:
  1. Redis hot cache        (key: user_vector:{id}, binary float32)
  2. Postgres persistent    (user_online_vectors table, survives restart)
  3. ALS base embedding     (from offline training, if user was in training set)
  4. Cold start             (mean of all user embeddings, for new users)
      │
      ▼
apply_feedback_update(u, v_movie, preference)
  u_new = u + η · preference · v_movie    (η = 0.05)
  if ‖u_new‖ > cap: u_new *= cap / ‖u_new‖  (cap = 10.0)
      │
      ▼
  → Redis: user_vector:{id}         (hot cache, immediate)
  → Postgres: user_online_vectors   (async fire-and-forget, upsert on conflict)
  → Redis: seen:user:{id}           (set, persisted across restarts)
  → Neo4j beacon map                (async fire-and-forget, max 3 concurrent writes)
```

### GET /feed → serve next movie

```
GET /api/v1/movies/feed
      │
      ▼
FeedManager.get_next_movie(user_id, user_preferences)
  1. lpop from Redis queue (feed:user:{id})
  2. if empty → refill_queue() (blocking)
     if below threshold (5) → refill_queue() (background task)
          │
          ▼
       refill_queue(user_id, user_preferences)
         fetch_n = batch_size (15) [× over_fetch_factor (2) if filters active]
         loop until queue filled or max_candidates (300) exhausted:
           Recommender.get_top_n_recommendations(user_id, n=fetch_n)
             scores = movie_embeddings[candidates] @ user_vector
             exclude seen → rank by score → return movie_ids
           hydrate new candidates with TMDB (parallel asyncio.gather)
           filter by user_preferences (_matches_preferences)
           if items found → break
           else fetch_n *= 2, retry with deeper ranked list
         push [movie_id, title] JSON entries to Redis queue
  3. mark movie as seen: sadd seen:user:{id}
  4. attach KG explanation (100ms timeout, silent fail)
  5. return MovieDetails
```

### Neo4j knowledge graph (explainability layer)

```
On each swipe (fire-and-forget, semaphore max 3 concurrent):
  update_beacon_on_swipe()
    query Neo4j: Movie → Director/Actor/Genre/Writer/Keyword
    update Redis beacon map:
      weight += swipe_score · entity_multiplier · recency_decay
    TTL: 24h

On feed request (100ms budget):
  explain_recommendation()
    load beacon map from Redis (or rebuild from Postgres + Neo4j)
    find_explanation_paths(movie_tmdb_id, beacon_map)
    score paths → render best as text
    attach to MovieDetails.explanation
```

Entity weights: Director×1.5, Actor×1.0, Genre×0.8, Writer×0.7, Keyword×0.5
Recency decay: 0.95 per day

All Neo4j calls are resilient — connection failures return empty results,
never propagate to the API response.

---

## Data flow summary

```
MovieLens CSV + Postgres swipes (offline)
      ↓
ALS embeddings (user_embeddings.npy, movie_embeddings.npy, mappings.json)
      ↓
Recommender loads on startup via load_model_artifacts()
      ↓
per-swipe : online vector update → Redis + Postgres (upsert)
per-feed  : dot product → ranked list → Redis queue → TMDB hydration → response
```

Item-CF artifacts are currently generated and evaluated offline only; online serving
still reads ALS artifacts.
