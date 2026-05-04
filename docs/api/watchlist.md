# Watchlist

Source: [backend/src/movie_recommender/api/v1/watchlist.py](../../backend/src/movie_recommender/api/v1/watchlist.py)

Save-for-later list. Independent of likes/dislikes — adding to the watchlist does **not** create a swipe and does **not** affect recommendations.

| Method | Path | Auth | Returns |
|---|---|---|---|
| POST | `/api/v1/watchlist/{movie_id}` | required | `WatchlistAddResponse` |
| DELETE | `/api/v1/watchlist/{movie_id}` | required | `WatchlistRemoveResponse` |
| GET | `/api/v1/watchlist?limit=&offset=` | required | `PaginatedMovieDetails` |

`{movie_id}` is the **internal numeric DB id**, not the TMDB id.

---

## `POST /watchlist/{movie_id}`

Add a movie to the user's watchlist.

**Response:** `{"movie_id": 42, "added": true}`

**Errors:**
- **404** `"User not found"` — auth user has no DB row
- **404** `"Movie not found"` — `{movie_id}` does not exist in `movies`
- **409** `"Movie already in watchlist"` — already on the list

---

## `DELETE /watchlist/{movie_id}`

Remove a movie from the user's watchlist.

**Response:** `{"movie_id": 42, "removed": true}`

**Errors:** **404** `"Movie not in watchlist"` if there's nothing to remove.

---

## `GET /watchlist`

Paginated list of full `MovieDetails`.

**Query:** `limit` (default 20), `offset` (default 0).

**Response — `PaginatedMovieDetails`:**
```jsonc
{ "items": [<MovieDetails>, ...], "total": 12, "limit": 20, "offset": 0 }
```
