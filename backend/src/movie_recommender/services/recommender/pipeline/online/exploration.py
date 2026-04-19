EXPLORATION_GENRE_KEY_PREFIX = "explore:genres:user:"
EXPLORATION_STATE_TTL_SECONDS = 7 * 24 * 60 * 60


def genre_impression_key(user_id: int) -> str:
    return f"{EXPLORATION_GENRE_KEY_PREFIX}{user_id}"


async def record_genre_impressions(
    redis_client,
    user_id: int,
    genres: list[str],
    ttl_seconds: int = EXPLORATION_STATE_TTL_SECONDS,
) -> None:
    """Track how often each genre has been served to a user."""
    if not genres:
        return

    key = genre_impression_key(user_id)
    unique_genres = {genre for genre in genres if genre}
    if not unique_genres:
        return

    for genre in unique_genres:
        await redis_client.hincrby(key, genre, 1)

    await redis_client.expire(key, ttl_seconds)


async def get_genre_impression_counts(redis_client, user_id: int) -> dict[str, int]:
    """Load per-genre impression counts for a user from Redis."""
    raw_counts = await redis_client.hgetall(genre_impression_key(user_id))
    if not raw_counts:
        return {}

    counts: dict[str, int] = {}
    for raw_genre, raw_count in raw_counts.items():
        genre = (
            raw_genre.decode("utf-8")
            if isinstance(raw_genre, (bytes, bytearray))
            else str(raw_genre)
        )
        counts[genre] = int(raw_count)

    return counts
