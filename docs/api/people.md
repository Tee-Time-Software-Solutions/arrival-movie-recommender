# People

Source: [backend/src/movie_recommender/api/v1/people.py](../../backend/src/movie_recommender/api/v1/people.py)

Knowledge-graph person lookups. Used by the Profile page's "see all the movies you liked with this director/actor" affordance.

| Method | Path | Auth | Returns |
|---|---|---|---|
| GET | `/api/v1/people/{person_tmdb_id}/linked-movies` | required | `List[LinkedMovie]` |

---

## `GET /people/{person_tmdb_id}/linked-movies`

Return the **current user's** liked movies that this person appears in. The result is filtered against the user's like history, not the global movie catalog — so it's empty if the user hasn't liked anything featuring this person.

`{person_tmdb_id}` is the TMDB person id (e.g. `1245` for Scarlett Johansson). It is not the internal DB id.

**Response item — `LinkedMovie`:**
```json
{ "tmdb_id": 949, "title": "Heat", "poster_url": "https://..." }
```

**Errors:** **404** `"User not found"` if the user row is missing. There's no 404 for "person has no linked movies" — it just returns `[]`.

---

## How the lookup works

`get_person_linked_movies` traverses the Neo4j beacon map: it starts from the user's beacon node, follows `LIKED` edges out to movies, then keeps movies that have an edge to the requested person. So this endpoint also returns `[]` if the KG hasn't been hydrated for any of the user's liked movies — common right after onboarding.
