# Chatbot

Source: [backend/src/movie_recommender/api/v1/chatbot.py](../../backend/src/movie_recommender/api/v1/chatbot.py)

This is the only endpoint that does **not** return JSON. It returns a `text/event-stream` body driven by `sse-starlette`. Clients should use `EventSource` or a fetch-based SSE parser (the frontend uses the latter — see [frontend/app/src/services/api/chat.ts](../../frontend/app/src/services/api/chat.ts)).

For deeper agent internals (the LangGraph wiring, tools, how to add a new tool), see [../chatbot_agent/langgraph_agent.md](../chatbot_agent/langgraph_agent.md).

| Method | Path | Auth | Body | Response |
|---|---|---|---|---|
| POST | `/api/v1/chatbot/stream` | required | `ChatRequest` | `text/event-stream` |

---

## Request — `ChatRequest`

```jsonc
{
  "message": "Recommend me a thriller from the 90s",
  // optional: prior turns; the frontend caps it at the last 10 messages
  "history": [
    { "role": "user", "content": "Hi" },
    { "role": "assistant", "content": "Hey! What kind of movie are you in the mood for?" }
  ]
}
```

`role` is one of `"user"` or `"assistant"`.

---

## SSE event taxonomy

| `event` | `data` shape | Emitted when |
|---|---|---|
| `token` | `{"token": "<text-chunk>"}` | LLM streams a chunk |
| `movies` | `{"movies": [<MovieDetails>, ...]}` | `search_movies` tool returns a non-empty list |
| `taste_profile` | `{"profile": {...}}` | `get_taste_profile` tool completes |
| `done` | `{}` | Stream end (success). Final event. |
| `error` | `{"error": "<message>"}` | Any exception. Final event on failure. |

Empty `search_movies` results (literal string `"No movies found matching those criteria."`) do **not** emit `event: movies` — the LLM still sees the text and adapts its reply.

### Wire format

```
event: token
data: {"token": "I'd recommend"}

event: token
data: {"token": " Heat (1995)"}

event: movies
data: {"movies": [{"movie_db_id":42,"tmdb_id":949,"title":"Heat", ...}]}

event: done
data: {}
```

Block separator is a blank line (`\n\n`). The frontend parser also tolerates `\r\n\r\n`. Each `data:` line is a single-line JSON document — no multi-line `data:` continuations are emitted.

### Ordering guarantees

- Tokens stream interleaved with tool events (a tool call may happen mid-reply).
- `done` is **always** the final event on success.
- `error` is the final event on failure — there will be no `done` after it.
- HTTP status of the stream itself is always **200** as long as auth + user lookup succeed. Transport-level errors are surfaced *inside* the stream as `event: error`.

---

## Authentication

Same Firebase Bearer token as every other authenticated endpoint. The decoded `uid` is used to look up the internal numeric `user_id`, which is then bound to both tools as a closure (see the agent doc for the security implications). If the user row doesn't exist, the endpoint returns **404** `{"detail": "User not found"}` *before* opening the stream.

---

## Example (curl)

```bash
curl -N -H "Authorization: Bearer <TOKEN>" \
     -H "Content-Type: application/json" \
     -d '{"message":"give me a feel-good movie"}' \
     http://localhost:8000/api/v1/chatbot/stream
```

`-N` disables curl's output buffering so you see tokens stream as they arrive.

---

## Common errors

| Status | Body | Cause |
|---|---|---|
| 401 | `{"detail":"Authentication failed"}` | Missing/invalid token |
| 404 | `{"detail":"User not found"}` | Token valid, but no `users` row matches its `uid` (e.g. user never called `/users/register`) |
| 422 | `{"detail":[...]}` | `message` missing or wrong type |
| 200 + `event: error` | `{"error":"..."}` | LLM/agent runtime error mid-stream (OpenRouter outage, tool exception, etc.) |
