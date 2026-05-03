# Users

Source: [backend/src/movie_recommender/api/v1/users.py](../../backend/src/movie_recommender/api/v1/users.py)

User profile, preferences, registered users, and KG-derived "top people". Most routes here use `verify_user(user_private_route=True)` — the `{user_id}` in the path **must equal the token's `uid`**, otherwise **403**.

> **Note:** `{user_id}` in these routes is the Firebase `uid` (string), not the internal numeric DB id.

| Method | Path | Auth | Body / Query | Returns |
|---|---|---|---|---|
| POST | `/api/v1/users/register` | Y | `UserCreate` | `UserCreatedResponse` |
| GET | `/api/v1/users/{user_id}/summary` | Y (own) | — | `UserProfileSummary` |
| GET | `/api/v1/users/{user_id}/liked-movies` | Y (own) | `limit, offset` | `PaginatedMovieDetails` |
| GET | `/api/v1/users/{user_id}/rated-movies` | Y (own) | `limit, offset` | `PaginatedMovieDetails` |
| PATCH | `/api/v1/users/{user_id}/preferences` | Y (own) | `UserPreferences` | `UserPreferences` |
| GET | `/api/v1/users/{user_id}/top-people` | Y (own) | `limit (default 5)` | `TopPeopleResponse` |

---

## `POST /users/register`

Create the internal `users` row after a Firebase signup. The Firebase token must already be valid.

**Body — `UserCreate`:**
```json
{ "firebase_uid": "abc123", "profile_image_url": "https://...", "email": "you@example.com" }
```

**Errors:** **409** if `firebase_uid` is already registered.

---

## `GET /users/{user_id}/summary`

Profile + analytics + preferences in one shot. Used by the Profile page.

Returns `UserProfileSummary`:
```jsonc
{
  "profile":     { "username": "...", "avatar_url": "...", "joined_at": "..." },
  "stats":       { "total_swipes": 0, "total_likes": 0, "total_dislikes": 0,
                   "total_seen": 0, "top_genres": [...] },
  "preferences": { "included_genres": [...], "excluded_genres": [...],
                   "min_release_year": null, "max_release_year": null,
                   "min_rating": null, "include_adult": false,
                   "movie_providers": [...] }
}
```

---

## `GET /users/{user_id}/liked-movies` / `rated-movies`

Paginated lists of full `MovieDetails`.

- `liked-movies` — only `like` swipes, most recent first.
- `rated-movies` — both likes and dislikes, ordered by preference score.

**Query:** `limit` (default 20), `offset` (default 0).

**Response — `PaginatedMovieDetails`:**
```jsonc
{ "items": [<MovieDetails>, ...], "total": 123, "limit": 20, "offset": 0 }
```

---

## `PATCH /users/{user_id}/preferences`

Update filters. Body is a full `UserPreferences` — replace, not merge. After a successful PATCH, also call `DELETE /movies/feed` so the queue rebuilds under the new filters (see [movies.md](movies.md)).

```json
{
  "included_genres": ["Sci-Fi"],
  "excluded_genres": ["Horror"],
  "min_release_year": 2000,
  "max_release_year": null,
  "min_rating": 7.0,
  "include_adult": false,
  "movie_providers": []
}
```

---

## `GET /users/{user_id}/top-people`

Top directors / actors / writers from the user's KG beacon map. Used by the Profile page's "your top people" panel.

**Query:** `limit` (default 5).

**Response — `TopPeopleResponse`:**
```jsonc
{
  "directors": [<TopPerson>, ...],
  "actors":    [<TopPerson>, ...],
  "writers":   [<TopPerson>, ...]
}
```

Each `TopPerson` carries `tmdb_id`, `name`, `weight`, `image_url`, and a small list of `linked_movies` (the user's liked films featuring this person).
