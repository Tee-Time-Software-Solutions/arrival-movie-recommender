# Movies

Source: [backend/src/movie_recommender/api/v1/movies.py](../../backend/src/movie_recommender/api/v1/movies.py)

The recommendation feed. Movies are pre-fetched into a per-user Redis queue by the recommender pipeline; these endpoints are the consumer side. For the *swipe* side (registering a like/dislike/skip), see [interactions.md](interactions.md).

| Method | Path | Auth | Returns |
|---|---|---|---|
| GET | `/api/v1/movies/feed/batch` | required | `List[MovieDetails]` |
| DELETE | `/api/v1/movies/feed` | required | `{"flushed": true}` |

---

## `GET /movies/feed/batch`

Pull the next *N* movies from the user's Redis feed queue. The handler resolves the user's preferences (genre filters, year range, minimum rating, adult flag) and asks `FeedManager.get_next_movie()` to honour them.

### Query

| Param | Type | Default | Range |
|---|---|---|---|
| `count` | int | 5 | 1–20 |

### Response — `List[MovieDetails]`

Full movie hydrate: id, TMDB id, title, poster, year, rating, genres, synopsis, cast, trailer, runtime, providers, KG metadata (keywords, collection, production companies), and an optional `explanation` (why this movie was recommended).

### Errors

| Status | Body | Cause |
|---|---|---|
| 401 | `{"detail":"Authentication failed"}` | Missing/invalid token |
| 404 | `{"detail":"User not found"}` | Token valid, but no `users` row matches its `uid` |
| 404 | `{"detail":"No movies found"}` | Feed queue empty AND the recommender produced nothing — usually means the user is brand new (run onboarding) or the recommender artifacts haven't loaded |

### Example

```bash
curl -H "Authorization: Bearer <TOKEN>" \
     "http://localhost:8000/api/v1/movies/feed/batch?count=5"
```

---

## `DELETE /movies/feed`

Drop everything in the user's Redis queue. Call this after the user changes preferences (genre filter, year range, etc.) so the next `/feed/batch` call rebuilds candidates from scratch under the new filters.

### Response

```json
{ "flushed": true }
```

Idempotent — calling it on an empty queue is a no-op.

### When to call from the frontend

Right after a successful `PATCH /users/{user_id}/preferences`. Otherwise the user keeps seeing movies generated under the *old* filters until the queue drains.
