# Single User Recommendation Journey — Test Summary

**Test:** `TestSingleUserJourney::test_user_recommendation_journey`
**Date:** 2026-02-19
**Status:** PASSED

---

## Overview

This integration test demonstrates how a single user (User 1) moves through the full recommendation pipeline — from cold-start recommendations to personalized results shaped by swipe interactions. The online updater adjusts the user's embedding vector in real time, and `get_top_n()` returns updated recommendations after each swipe.

### Test Configuration

| Parameter | Value |
|-----------|-------|
| Embedding dimensions | 6 (2 per genre) |
| Genre mapping | ACTION → dims [0-1], COMEDY → dims [2-3], HORROR → dims [4-5] |
| Learning rate (eta) | 0.3 |
| Movies | 12 (4 action, 4 comedy, 4 horror) |
| Recommendations per step | 5 |

---

## Step-by-Step Journey

### Step 1 — Cold Start (no history)

User opens the app with no swipe history. Recommendations are based solely on the initial user vector, which has a slight action lean.

| Rank | Movie | Genre |
|------|-------|-------|
| 1 | Die Hard | ACTION |
| 2 | John Wick | ACTION |
| 3 | Mad Max | ACTION |
| 4 | Top Gun | ACTION |
| 5 | Bridesmaids | COMEDY |

> Initial vector already favors action — 4 of 5 recs are action films.

---

### Step 2 — LIKE "Die Hard" (ACTION)

| Rank | Movie | Genre |
|------|-------|-------|
| 1 | John Wick | ACTION |
| 2 | Mad Max | ACTION |
| 3 | Top Gun | ACTION |
| 4 | Bridesmaids | COMEDY |
| 5 | Superbad | COMEDY |

| Genre | Avg Score Delta |
|-------|-----------------|
| ACTION | **+0.2750** |
| COMEDY | +0.0075 |
| HORROR | +0.0000 |

> Action scores jump significantly. Die Hard exits recs (already seen). Comedy starts appearing as action slots fill up with seen movies.

---

### Step 3 — LIKE "John Wick" (ACTION, double down)

| Rank | Movie | Genre |
|------|-------|-------|
| 1 | Mad Max | ACTION |
| 2 | Top Gun | ACTION |
| 3 | Bridesmaids | COMEDY |
| 4 | Superbad | COMEDY |
| 5 | The Hangover | COMEDY |

| Genre | Avg Score Delta |
|-------|-----------------|
| ACTION | **+0.2550** |
| HORROR | +0.0277 |
| COMEDY | +0.0075 |

> Second action like reinforces the action dimensions. More comedy fills in as action movies get consumed.

---

### Step 4 — DISLIKE "The Shining" (HORROR)

| Rank | Movie | Genre |
|------|-------|-------|
| 1 | Mad Max | ACTION |
| 2 | Top Gun | ACTION |
| 3 | Bridesmaids | COMEDY |
| 4 | Superbad | COMEDY |
| 5 | The Hangover | COMEDY |

| Genre | Avg Score Delta |
|-------|-----------------|
| ACTION | +0.0000 |
| COMEDY | +0.0000 |
| HORROR | **-0.2750** |

> Horror scores drop sharply. Action and comedy unaffected — the orthogonal embeddings ensure genre independence.

---

### Step 5 — SKIP "Bridesmaids" (COMEDY)

| Rank | Movie | Genre |
|------|-------|-------|
| 1 | Mad Max | ACTION |
| 2 | Top Gun | ACTION |
| 3 | Superbad | COMEDY |
| 4 | The Hangover | COMEDY |
| 5 | Mean Girls | COMEDY |

> **Vector unchanged.** Skip sends preference=0, so the online updater makes no adjustment. Bridesmaids is removed from recommendations (marked as seen) but no learning occurs.

---

### Step 6 — LIKE "Mad Max" (ACTION, third action like)

| Rank | Movie | Genre |
|------|-------|-------|
| 1 | Top Gun | ACTION |
| 2 | Superbad | COMEDY |
| 3 | The Hangover | COMEDY |
| 4 | Mean Girls | COMEDY |
| 5 | The Conjuring | HORROR |

| Genre | Avg Score Delta |
|-------|-----------------|
| ACTION | **+0.2340** |
| COMEDY | +0.0000 |
| HORROR | +0.0000 |

> Third action like continues boosting action. With most action movies seen, comedy and even a horror film appear to fill the top 5.

---

## Journey Summary

| Metric | Value |
|--------|-------|
| Total swipes | 5 |
| Action LIKES | 3 |
| Horror DISLIKES | 1 |
| Comedy SKIPS | 1 |
| Movies seen | 5 |
| Recs returned per step | 5 |

### User Vector Evolution

| Dimension | Start | Final | Change |
|-----------|-------|-------|--------|
| ACTION [dim 0] | +0.4000 | +1.2700 | **+0.8700** |
| ACTION [dim 1] | +0.1000 | +0.1900 | +0.0900 |
| COMEDY [dim 2] | +0.2000 | +0.2000 | 0.0000 |
| COMEDY [dim 3] | +0.1000 | +0.1000 | 0.0000 |
| HORROR [dim 4] | +0.1000 | -0.1700 | **-0.2700** |
| HORROR [dim 5] | +0.0000 | -0.0300 | -0.0300 |

**Vector drift (L2 norm):** 0.9159
**Online vector stored:** Yes

### Key Observations

1. **Likes shift the vector toward the liked genre** — each action LIKE increased action dimension scores by ~0.25 on average
2. **Dislikes shift away from the disliked genre** — the horror DISLIKE dropped horror scores by -0.275
3. **Skips have zero effect** — preference=0 means no vector update
4. **Genre independence is preserved** — liking action does not meaningfully affect comedy or horror scores (orthogonal embeddings)
5. **Seen movies are excluded** — recommendations never repeat a swiped movie
6. **As preferred movies are consumed, other genres fill in** — after 3 action movies are seen, comedy and horror appear in top 5

---

## Technical Notes

- Embeddings are hand-crafted with orthogonal genre dimensions (not ALS-trained) to ensure clear, interpretable genre separation in test output
- Production uses 64-dim ALS embeddings trained on MovieLens-25M; the same `update_user_vector()` and `get_top_n()` functions are used in both contexts
- The online updater formula: `user_vector += eta * preference * movie_vector`, with norm capping to prevent explosion
