# Recommender — Architecture

## Offline pipeline (ALS)

Run once to produce embeddings. Re-run when retraining.

```
data/raw/
  movies.csv
  ratings.csv
      │
      ▼
preprocess_movies()                         [preprocess_movies.py]
  extract year, clean title, split genres
  → data/processed/movies_clean.parquet

preprocess_ratings()                        [preprocess_ratings.py]
  rating (0.5–5.0)
  → bucket {1,2,3,4}
  → preference {-2,-1,+1,+2}
  → data/processed/interactions_clean.parquet
      │
      ▼
filtering()                                 [filtering.py]
  iterative core filter until stable:
    remove movies  < 20 interactions
    remove users   < 10 interactions
  26k movies → ~13k | 138k users remain
  → data/processed/interactions_filtered.parquet

prune_movies()                              [prune_movies.py]
  keep only movies that survived filtering
  → data/processed/movies_filtered.parquet
      │
      ▼
split()                                     [split.py]
  per-user chronological split (no leakage)
  80% train / 10% val / 10% test
  → data/splits/train.parquet
  → data/splits/val.parquet
  → data/splits/test.parquet
      │
      ▼
build_sparse_matrix()                       [build_matrix.py]
  CSR matrix R_train (users × movies)
  values = preference score
  + contiguous ID mappings
  → artifacts/R_train.npz
  → artifacts/mappings.json
      │
      ▼
train()                                     [train_als.py]
  confidence: C = 1 + α·|preference|, α=15
  implicit.ALS(factors=64, reg=0.1, iters=15)
  → artifacts/user_embeddings.npy   shape (138k, 64)
  → artifacts/movie_embeddings.npy  shape (13k, 64)
  → artifacts/model_info.json
      │
      ▼
evaluate()                                  [evaluate.py]
  for each val user:
    scores = movie_embeddings @ user_vector
    mask seen movies → take top 10
    compute Precision@10, Recall@10, NDCG@10
  baseline results: P=0.055, R=0.081, NDCG=0.183
```

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
current_user_vector(artifacts, online_cache, user_id)
  known user     → artifacts.user_embeddings[index]   ← from offline training
  new user       → mean(all user embeddings)           ← cold start
  returning user → online_user_vectors[user_id]        ← overrides base
      │
      ▼
update_user_vector(u, v_movie, preference, η=0.05, cap=10.0)
  u_new = u + η · preference · v_movie
  if ‖u_new‖ > cap: u_new *= cap / ‖u_new‖
      │
      ▼
store → online_user_vectors[user_id]   (in-process, lost on restart)
store → Redis seen set                 (persisted, survives restart)
fire  → Neo4j beacon update            (async, fire-and-forget)
```

### GET /feed → serve next movie

```
GET /api/v1/movies/feed
      │
      ▼
FeedManager.get_next_movie(user_id)
  1. load seen set from Redis
  2. pop from Redis queue (feed:user:{id}), skip seen movies
  3. if queue empty or below threshold → refill_queue()
          │
          ▼
       Recommender.get_top_n_recommendations(user_id, all_movie_ids)
         scores = movie_embeddings[candidates] @ user_vector  ← dot product
         filter seen → rank → return ordered movie_ids
          │
          ▼
       hydrate top N with TMDB (async, parallel)
       push JSON [movie_id, title] entries to Redis queue
  4. attach KG explanation (100ms timeout, silent fail if missing)
  5. return MovieDetails
```

### Neo4j knowledge graph (explainability layer)

```
On each swipe (fire-and-forget):
  update_beacon_on_swipe()
    query Neo4j: Movie → Director/Actor/Genre/Writer/Keyword
    update Redis beacon map:
      weight += swipe_score · entity_multiplier · recency_decay
    TTL: 24h

On feed request (100ms budget):
  explain_recommendation()
    load beacon map from Redis (or rebuild from DB + Neo4j)
    find_explanation_paths(movie_tmdb_id, beacon_map)
    score paths → render best as text
    attach to MovieDetails.explanation
```

Entity weights: Director×1.5, Actor×1.0, Genre×0.8, Writer×0.7, Keyword×0.5
Recency decay: 0.95 per day

**Requirement:** Neo4j must be seeded via `scripts/seed_kg.py` before explanations work.
If the graph is empty, the system silently serves recommendations without explanations.

---

## Data flow summary

```
MovieLens 20M (offline)
      ↓
embeddings (user_embeddings.npy, movie_embeddings.npy)
      ↓
Recommender loads on startup (artifact_loader.py)
      ↓
per-swipe: online_user_vectors updated in-process
per-feed:  dot product → ranked list → Redis queue → TMDB hydration → response
```
