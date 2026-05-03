# Onboarding

Source: [backend/src/movie_recommender/api/v1/onboarding.py](../../backend/src/movie_recommender/api/v1/onboarding.py)

First-run flow: show the user a curated grid of popular movies, let them search TMDB live, then submit their selections to bootstrap their recommender vector.

| Method | Path | Auth | Returns |
|---|---|---|---|
| GET | `/api/v1/onboarding/movies` | required | `List[OnboardingMovieCard]` |
| GET | `/api/v1/onboarding/search?query=...` | required | `List[OnboardingSearchResult]` |
| POST | `/api/v1/onboarding/complete` | required | `OnboardingCompleteResponse` |

---

## `GET /onboarding/movies`

Returns ~30 curated popular movies (sampled across genres) for the onboarding grid.

**Errors:** **503** `"Onboarding movies not yet seeded. Run the seed command first."` — the bootstrap pool hasn't been populated. Not a runtime concern; the seeding task (`seed_onboarding_movies`) runs on app startup.

---

## `GET /onboarding/search?query=...`

Live TMDB search for users who don't see what they want in the curated grid.

**Query:** `query: str` (must be ≥ 2 non-whitespace chars; otherwise returns `[]`).

**Response item — `OnboardingSearchResult`:**
```json
{
  "movie_db_id": 42,        // null if the movie isn't in our DB yet
  "tmdb_id": 949,
  "title": "Heat",
  "poster_url": "https://image.tmdb.org/...",
  "release_year": 1995
}
```

**Errors:** **502** if the upstream TMDB call fails.

---

## `POST /onboarding/complete`

Submit the chosen movies and finish onboarding. Bulk-creates `like` swipes, marks the user as onboarded, and seeds the recommender vector.

**Body — `OnboardingSubmission`:**
```json
{ "movie_db_ids": [12, 34, 56, 78, 90] }
```

Constraint: **5–33 items**. Movies that don't exist in our DB are silently skipped (the hydrator may also have to fetch metadata before the swipe lands).

**Response:** `{"onboarding_completed": true}`

**Errors:** **404** if `users` row missing; **409** if onboarding was already completed.

---

## What "complete" actually does

1. Hydrate each `movie_db_id` (fetch TMDB metadata if not already enriched).
2. `create_swipes_bulk(user_id, movie_db_ids, "like")` — writes one row per movie to `swipes`.
3. `mark_onboarding_completed(user_id)` → `users.onboarding_completed = true`.
4. Add all movie ids to the user's Redis "seen" set so they don't reappear in the feed.
5. For each movie, call `recommender.set_user_feedback(...)` — same path as a normal like swipe — to initialise the user vector.

After this returns, `/movies/feed/batch` should produce real recommendations.
