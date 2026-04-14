# Recommender — Set-up

## 1. Get the dataset

Download **MovieLens (small)** from https://grouplens.org/datasets/movielens/latest/
Note: you can use the big version that contain 10M, but need to update from small extension in the file paths to big.

Place only these two files here:

```
backend/src/movie_recommender/services/recommender/pipeline/artifacts/dataset/source/small/
  movies.csv
  ratings.csv
```

Create the folder if it doesn't exist:
```bash
mkdir -p backend/src/movie_recommender/services/recommender/pipeline/artifacts/dataset/source/small
```

> Paths are configured in `config.yaml` (see step 4). The `small` dataset is enough for local dev.

---

## 2. Install dependencies

From repo root:
```bash
cd backend && uv sync
```

---

## 3. Run the offline pipeline

From the repo root with the backend venv active:

```bash
cd backend
uv run python -m movie_recommender.services.recommender.pipeline.offline.models.als.main
```

Run Item-CF (offline-only artifacts + metrics):

```bash
cd backend
uv run python -m movie_recommender.services.recommender.pipeline.offline.models.item_cf.main
```

This runs the 10-step pipeline:

| Step | What it does | Output |
|------|-------------|--------|
| 1 | Preprocess movies | `processed/movies_clean.parquet` |
| 2 | Preprocess ratings | `processed/ratings_clean.parquet` |
| 3 | Fetch app swipes from Postgres | `raw/swipes_from_db.parquet` |
| 4 | Merge MovieLens + app swipes | `processed/ratings_unified.parquet` |
| 5 | Filter sparse users/movies | `processed/ratings_filtered.parquet` |
| 6 | Prune movies to interaction set | `processed/movies_filtered.parquet` |
| 7 | Chronological train/val/test split | `splits/train.parquet` etc. |
| 8 | Build sparse CSR matrix | `model_assets/R_train.npz`, `mappings.json` |
| 9 | Train ALS | `model_assets/user_embeddings.npy`, `movie_embeddings.npy` |
| 10 | Evaluate | `model_assets/als_metrics.json` |

Item-CF uses the same first 7 base steps, then writes:
- `model_assets/item_cf_train_matrix.npz`
- `model_assets/item_cf_mappings.json`
- `model_assets/item_cf_similarity.npz`
- `model_assets/item_cf_model_info.json`
- `model_assets/item_cf_metrics.json`

All artifacts land under:
```
backend/src/movie_recommender/services/recommender/pipeline/artifacts/
  dataset/
    source/small/      ← place MovieLens CSVs here
    raw/               ← swipes_from_db.parquet (auto-generated)
    processed/
    splits/
  model_assets/
```

**To skip the Postgres swipe export** (no DB needed):
```bash
SKIP_DB_SWIPE_EXPORT=1 uv run python -m movie_recommender.services.recommender.pipeline.offline.models.als.main
```

Item-CF supports the same `SKIP_DB_SWIPE_EXPORT=1` behavior.

Online serving still uses ALS artifacts for live requests.

Expected runtime on M1 (small dataset): ~2–5 min.

---

## 4. Pipeline config

All paths and hyperparameters live in:
```
backend/src/movie_recommender/services/recommender/config.yaml
```

Paths are relative to the `recommender/` root. Override with absolute paths if needed.

```yaml
data_dirs:
  source_dir: "pipeline/artifacts/dataset/source/small"
  processed_dir: "pipeline/artifacts/dataset/processed"
  splits_dir: "pipeline/artifacts/dataset/splits"
  model_assets_dir: "pipeline/artifacts/model_assets"

models:
  als:
    factors: 16
    regularization: 0.1
    iterations: 15
    alpha: 15        # C(u,i) = 1 + alpha * |preference|
  item_cf:
    similarity: "cosine"
    top_k_neighbors: 100
    min_similarity: 0.0
    use_positive_only: true
    normalize_scores: true
```

---

## 5. Run unit tests (no dataset needed)

```bash
cd backend
uv run pytest tests/unit/ -v
```

Uses synthetic in-memory fixtures — no data on disk required.

---

## 6. Nightly rerun (cron)

`run_pipeline_cron_job()` in `pipeline/offline/models/als/main.py` is registered with APScheduler at midnight.
It acquires a Redis lock (`ml_pipeline_lock`, TTL 1h) before running and releases it in `finally`.
No manual action needed — runs automatically when the backend is up.
