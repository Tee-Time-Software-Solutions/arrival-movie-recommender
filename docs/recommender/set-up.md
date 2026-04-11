# Recommender — Set-up

## 1. Get the dataset

Download **MovieLens 20M** from https://grouplens.org/datasets/movielens/20m/

Place only these two files here:

```
backend/src/movie_recommender/services/recommender/data/raw/movies.csv
backend/src/movie_recommender/services/recommender/data/raw/ratings.csv
```

Create the folder if it doesn't exist:
```bash
mkdir -p backend/src/movie_recommender/services/recommender/data/raw
```

## 2. Install dependencies

From repo root:
```bash
cd backend && uv sync
```

## 3. Run the offline pipeline (ALS only — FM is broken)

From repo root, with the backend venv active:
```bash
cd backend
uv run python src/movie_recommender/services/recommender/learning/offline_pipelines/implicit_als.py
```

This runs the full 8-step pipeline and writes artifacts to:
```
backend/src/movie_recommender/services/recommender/artifacts/
  movie_embeddings.npy
  user_embeddings.npy
  mappings.json
  R_train.npz
  model_info.json
```

And processed data to:
```
backend/src/movie_recommender/services/recommender/data/
  processed/
    movies_clean.parquet
    interactions_clean.parquet
    interactions_filtered.parquet
    movies_filtered.parquet
  splits/
    train.parquet
    val.parquet
    test.parquet
```

Expected runtime on M1: ~15–25 min (ratings preprocessing dominates).

## 4. FM pipeline — skip for now

`factorization_machines.py` is broken because `lightfm` can't be correctly installed.
Acknowledge in the report; ALS is the production serving model.

## 5. Run unit tests (no dataset needed)

```bash
cd backend
uv run pytest tests/unit/ -v
```

Uses synthetic in-memory fixtures. No data on disk required.

## 6. Known artifact paths

All paths are resolved relative to `paths_dev.py`:

```python
PROJECT_ROOT = backend/src/movie_recommender/services/recommender/
DATA_RAW      = PROJECT_ROOT/data/raw
DATA_PROCESSED = PROJECT_ROOT/data/processed
DATA_SPLITS   = PROJECT_ROOT/data/splits
ARTIFACTS     = PROJECT_ROOT/artifacts
```
