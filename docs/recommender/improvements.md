# Recommender — Known Issues & Improvements

## Broken

**FM pipeline (lightfm) — blocked**
- `lightfm` fails to install correctly in the current environment
- `factorization_machines.py` cannot be run
- Plan: acknowledge in report, skip in demo, ALS is the live serving model

---

## Bugs / design problems

**1. `SEEN_KEY_PREFIX` is duplicated**
- Defined identically in `services/recommender/main.py:26` and `services/feed_manager/main.py:14`
- Risk: if one changes, they silently diverge and users see repeated movies
- Fix: move to a shared constants module

**2. `online_user_vectors` is not persisted**
- Lives in process memory only; Redis is used for seen-movie sets but not for user vectors
- Every backend restart wipes all personalization
- Fix: serialize vectors to Redis (`HSET user_vec:{user_id} ...`) on write, load on first access

**3. `FeedManager` reaches into `Recommender` internals**
- `feed_manager/main.py:143`: `self.recommender.artifacts.movie_id_to_index.keys()`
- Couples FeedManager to Recommender's internal data structure
- Fix: add `Recommender.get_all_candidate_ids() -> list[int]`

**4. `paths_dev.py` is used in production**
- The filename implies dev-only but is imported by all serving code
- Misleading; rename to `paths.py`

**5. `evaluate.py` duplicates the metric loop from `compare_als_vs_fm.py`**
- Both implement Precision/Recall/NDCG computation independently
- `_calculate_metrics()` in `compare_als_vs_fm.py` should live in `metrics.py` and be imported by both

**6. ALS pipeline doesn't skip existing artifacts**
- `implicit_als.py` always re-runs all 8 steps
- FM pipeline checks disk before each step and skips if artifact exists
- Fix: mirror the FM pattern in the ALS pipeline

**7. `user_id` type is inconsistent**
- Recommender methods accept `str` (see `user_vectors.py:to_int_user_id`)
- DB and API use `int`
- Works because of the conversion function, but fragile — any caller passing an `int` silently gets cold-start for new Firebase UID users

---

## Planned additions

**Third model — popularity baseline**
- `score(movie) = interaction count in train set`
- No ML, just counting → establishes the floor every real model must beat
- Needed to claim the benchmark is rigorous (currently ALS vs FM only)
- File: `learning/offline_pipelines/popularity.py`
- Extend `compare_als_vs_fm.py` → `compare_models.py`

**Hyperparameter search (low priority)**
- Current ALS: factors=64, α=15, reg=0.1, iters=15
- Could improve NDCG@10 (0.183 baseline) with a grid search over factors and α
- Only worth doing if the embedding quality feels poor in the live demo
