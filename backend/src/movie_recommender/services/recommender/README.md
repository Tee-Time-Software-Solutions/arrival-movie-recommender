## How to run the offline pipeline

To fully reconstruct the offline recommender pipeline, you need to:

1. **Install and start the project (from repo root)**
   - `make install`
   - `make dev-start`

2. **Activate the backend virtualenv**
   - `cd backend`
   - `source .venv/bin/activate`

3. **Go to the recommender service**
   - `cd src/movie_recommender/services/recommender`

4. **Prepare the data directory**
   - Create the folders:
     - `data/`
     - `data/raw/`
   - Download the **MovieLens 20M** dataset and place **only** these two CSV files into `data/raw/`:
     - `movies.csv`
     - `ratings.csv`

5. **Run the full offline pipeline**
   - From `src/movie_recommender/services/recommender/`:
     - `cd learning`
     - `python offline_pipeline.py`

This will execute the whole offline pipeline end‑to‑end and regenerate all artifacts (filtered interactions, splits, sparse matrix, embeddings, and evaluation metrics).

---

## What the offline pipeline does (current state)

The code in this folder implements a complete **offline collaborative filtering pipeline** on MovieLens 20M, using **implicit ALS**.

- **Data preprocessing**
  - `preprocess_movies.py`: cleans and normalizes movie metadata.
  - `preprocess_ratings.py`: reads `ratings.csv`, maps raw ratings to:
    - bucket 1–4, then
    - symmetric preference scale \(-2, -1, +1, +2\).

- **Filtering**
  - `filtering.py` (`run_filtering`): iteratively removes sparse users/movies to reach a denser core:
    - drops movies with very few interactions,
    - keeps a stable, denser interaction matrix.

- **Temporal train/val/test split**
  - `split.py` (`run_split`): chronological **per‑user** split into:
    - 80% train, 10% validation, 10% test,
    - no temporal leakage (future interactions never appear in train).

- **Matrix construction**
  - `learning/build_matrix.py` (`build_sparse_matrix`):
    - builds contiguous ID mappings:
      - `user_id → user_index`
      - `movie_id → movie_index`
    - constructs a CSR matrix \(R_{\text{train}}\) of shape `(num_users, num_movies)`,
    - saves:
      - `R_train.npz`
      - `mappings.json` (ID/index dictionaries).

- **Model training (implicit ALS)**
  - `learning/train_als.py` (`train`):
    - loads `R_train.npz`,
    - converts ratings/preferences into a confidence matrix:
      \[
      C = 1 + \alpha \cdot |r|,\ \alpha = 15
      \]
    - trains an implicit ALS model with 64 latent factors,
    - saves:
      - `movie_embeddings.npy` (item factors),
      - `user_embeddings.npy` (user factors),
      - `model_info.json` (hyperparameters, dimensions).

- **Offline evaluation**
  - `learning/evaluate.py` (`evaluate`):
    - loads embeddings, mappings, `train.parquet` and `val.parquet`,
    - for each validation user:
      - scores all movies via dot product,
      - filters out movies seen in train,
      - ranks and takes top‑K (K=10),
    - computes:
      - Recall@K
      - Precision@K
      - NDCG@K
    - prints summary metrics for the current model.

Overall, you now have a **reproducible, end‑to‑end offline CF pipeline** with:
- consistent preprocessing,
- dense interaction matrix,
- properly trained ALS embeddings,
- and sound offline evaluation.

---

## Online learning / user‑vector updates (next steps)

The current system learns **static** user embeddings during offline training. To support **online learning**, you need to update user vectors in response to new user feedback (swipes / ratings) in the app.

At a high level, you can:

- **1. Represent preferences**
  - Keep using the existing 1–4 preference buckets (and their mapping to \(-2, -1, +1, +2\)), or
  - Work directly with a 1–4 scale if that’s what the frontend sends.

- **2. Maintain user vectors**
  - Start from the offline‑trained `user_embeddings` for existing users.
  - For new users (cold start), initialize a vector (e.g. zeros, small random values, or an average user vector).

- **3. Update rule for online feedback**
  - When a user \(u\) provides feedback on a movie \(i\) with preference \(p \in \{1,2,3,4\}\) (or mapped \(\tilde{p} \in \{-2,-1,1,2\}\)):
    - Let:
      - \(u\_v\) = current user embedding (vector),
      - \(m\_v\) = movie embedding from `movie_embeddings`,
      - \(\eta\) = small learning rate (e.g. 0.01–0.1).
    - A simple online update step is:
      \[
      u\_v \leftarrow u\_v + \eta \cdot \tilde{p} \cdot m\_v
      \]
    - Optionally, apply:
      - normalization (e.g. divide by \(\|u\_v\|\) to keep vector norms bounded), or
      - clipping to avoid explosions.

- **4. Where to integrate this**
  - The `Recommender` class that the backend communicates with lives in `main.py` at `backend/src/movie_recommender/services/recommender/main.py`.
  - To keep that file small, do the heavy work inside `backend/src/movie_recommender/services/recommender/serving`. Put in the `Recommender` class only what is strictly necessary (e.g. thin wrappers that delegate to serving logic).
  - The serving layer should:
    - load `movie_embeddings.npy`, `user_embeddings.npy`, and `mappings.json`,
    - on each new interaction:
      - map `(user_id, movie_id)` to indices,
      - apply the online update rule to the in‑memory user vector,
      - persist the updated user vector (e.g. in Redis or a database),
    - use the updated user vector to score movies (dot product) for future recommendations.

- **5. Guardrails**
  - Choose a small learning rate \(\eta\) to avoid overreacting to single swipes.
  - Optionally:
    - decay the contribution of older feedback,
    - limit the maximum norm of user vectors.

With this online update mechanism in place, recommendations will **adapt in real time** to each user’s latest preferences, while still leveraging the strong offline ALS model you have already trained.

Perfect moment to zoom out.

You’ve built something substantial. Let’s structure this clearly.

---

# 🏗 WHAT YOU HAVE ACCOMPLISHED

You have built a **complete offline collaborative filtering pipeline** from scratch.

Let’s break it down layer by layer.

---

# 1️⃣ Data Engineering Layer (Complete)

### ✔ Raw ingestion

* Loaded ML-20M safely (correct dtypes)
* Managed 20M interactions on local M1

### ✔ Preference transformation

* Converted decimal ratings → 1–4 buckets
* Converted buckets → symmetric preference scale (-2, -1, +1, +2)

This is critical for swipe-based modeling.

### ✔ Sparse filtering (iterative core pruning)

* Removed movies with <20 interactions
* Preserved user universe
* Reduced movie universe from 26k → 13k
* Stabilized training density

Professional-level preprocessing.

---

# 2️⃣ Proper Evaluation Protocol (Very Important)

You did this correctly:

### ✔ Chronological per-user split

* 80 / 10 / 10
* No leakage
* Simulates real deployment scenario

Most students get this wrong.

You didn’t.

---

# 3️⃣ Matrix Construction (Correct and Efficient)

### ✔ Created contiguous ID mappings

* user_id → user_index
* movie_id → movie_index

### ✔ Built CSR sparse matrix

* ~15.9M non-zero entries
* Correct shape: (138k users × 13k movies)

Efficient memory usage.

---

# 4️⃣ Model Training (Implicit ALS)

### ✔ Correct matrix orientation

### ✔ Proper confidence weighting

* C = 1 + α |r|
* α = 15

### ✔ Trained 64-dimensional embeddings

* Movie embeddings: (13131, 64)
* User embeddings: (138493, 64)

### ✔ Saved artifacts properly

* movie_embeddings.npy
* user_embeddings.npy
* mappings.json
* model_info.json

This is production-quality structure.

---

# 5️⃣ Offline Evaluation (Critical Milestone)

Metrics:

```
Recall@10    = 0.0808
Precision@10 = 0.0550
NDCG@10      = 0.1827
```

These are:

* Healthy
* Above random
* Correctly computed
* Using true embeddings

You now have a validated collaborative filtering model.

---

# 6️⃣ Full Reproducible Pipeline

You created:

```
offline_pipeline.py
```

One command:

```
python offline_pipeline.py
```

Rebuilds entire model.

This is exactly how serious ML systems operate.

---

# 🎯 WHAT YOU HAVE RIGHT NOW

You have:

✔ A working collaborative filtering engine
✔ A reproducible offline pipeline
✔ Clean artifacts for serving
✔ Correct evaluation protocol
✔ Structured project architecture

You have completed:

> Phase 1 — Offline Learning

---

# 🚧 WHAT IS MISSING

Now we move to Phase 2.

---

# 🟡 1️⃣ Serving Layer (Not Implemented Yet)

You still need to implement:

```
Recommender.get_top_n()
Recommender.update_user()
Recommender.similar_movies()
```

This will:

* Load artifacts
* Compute dot products
* Rank movies
* Integrate with backend

Right now you have no runtime scoring layer.

---

# 🟡 2️⃣ Online User Updates

Currently:

* User embeddings are static (from training)

But in your app:

* Users swipe in real-time
* Their embedding must update
* Recommendations must change dynamically

We need to implement:

```
user_vector += learning_rate * preference * movie_vector
```

Or similar controlled update rule.

---

# 🟡 4️⃣ Cold Start Strategy

New user:

* No interactions
* No embedding

We must define:

* Popular movie fallback
* Genre-based fallback

Not implemented yet.

---

# 🟡 5️⃣ Hyperparameter Tuning (Optional Improvement)

You could improve metrics by tuning:

* factors
* regularization
* alpha
* iterations

But this is enhancement, not missing functionality.

---

# 🟡 6️⃣ Integration With Backend Class

Your backend teammate expects:

```python
class Recommender:
    def get_top_n(...)
```

That bridge is not built yet.



