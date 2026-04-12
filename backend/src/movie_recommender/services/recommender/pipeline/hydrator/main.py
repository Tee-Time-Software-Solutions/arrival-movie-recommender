import asyncio
import logging
import re

import httpx

from movie_recommender.core.settings.main import AppSettings
from movie_recommender.schemas.requests.movies import (
    CastMember,
    MovieDetails,
    MovieProvider,
    TMDBKeyword,
    TMDBCollection,
    TMDBProductionCompany,
)
from movie_recommender.database.CRUD.movies import (
    get_movie_by_id,
    save_hydrated_movie,
    movie_to_details,
)

logger = logging.getLogger(__name__)

# Regex for cleaning movie titles from the MovieLens dataset
_AKA_RE = re.compile(r"\s*\(a\.k\.a\..*?\)")
_ARTICLE_SUFFIX_RE = re.compile(r"^(.+),\s*(The|A|An)$")
_YEAR_RE = re.compile(r"\s*\((\d{4})\)\s*$")


def _parse_ml_title(title: str) -> tuple[str, int | None]:
    """Parse a MovieLens-formatted title into (clean_title, year).

    'Prestige, The (2006)' → ('The Prestige', 2006)
    'Matrix, The (1999)'   → ('The Matrix', 1999)
    'Amélie (2001)'        → ('Amélie', 2001)
    """
    m = _YEAR_RE.search(title)
    year = int(m.group(1)) if m else None
    without_year = title[: m.start()].strip() if m else title
    without_aka = _AKA_RE.sub("", without_year)
    clean = _ARTICLE_SUFFIX_RE.sub(r"\2 \1", without_aka).strip()
    return clean, year


class TMDBFetcher:
    def __init__(self) -> None:
        settings = AppSettings()
        self.TMDB_API_KEY = settings.tmdb.api_key
        self.BASE_URL = settings.tmdb.base_url
        self.IMG_URL = settings.tmdb.img_url
        self._client = httpx.AsyncClient(timeout=10.0)

    async def search_movies(self, query: str) -> list[dict]:
        """Search TMDB for movies by title. Returns raw TMDB result items."""
        response = await self._client.get(
            f"{self.BASE_URL}/search/movie",
            params={"api_key": self.TMDB_API_KEY, "query": query},
        )
        return response.json().get("results", [])[:10]

    async def fetch_detail_by_id(self, tmdb_id: int) -> dict | None:
        """Fetch full movie detail from TMDB by ID. Returns None if not found."""
        try:
            res = (
                await self._client.get(
                    f"{self.BASE_URL}/movie/{tmdb_id}",
                    params={
                        "api_key": self.TMDB_API_KEY,
                        "append_to_response": "credits,videos,watch/providers,keywords",
                    },
                )
            ).json()
        except Exception:
            logger.warning(f"TMDB fetch failed for tmdb_id={tmdb_id}", exc_info=True)
            return None
        return res if "id" in res else None

    async def fetch_detail_by_title(self, movie_title: str) -> dict | None:
        """Search TMDB by title, then fetch full detail for the top result."""
        clean, year = _parse_ml_title(movie_title)
        params: dict = {"api_key": self.TMDB_API_KEY, "query": clean}
        if year:
            params["year"] = year
        search_res = (
            await self._client.get(f"{self.BASE_URL}/search/movie", params=params)
        ).json()
        if not search_res.get("results"):
            return None
        tmdb_id = search_res["results"][0]["id"]
        return await self.fetch_detail_by_id(tmdb_id)

    def parse_detail_response(self, movie_db_id: int, detail_res: dict) -> MovieDetails:
        """Build a MovieDetails from a raw TMDB /movie/{id} response dict."""
        poster_path = detail_res.get("poster_path")
        release_date = detail_res.get("release_date", "")
        return MovieDetails(
            movie_db_id=movie_db_id,
            tmdb_id=detail_res["id"],
            title=detail_res.get("original_title", ""),
            poster_url=f"{self.IMG_URL}{poster_path}" if poster_path else "",
            release_year=int(release_date[:4]) if release_date else 0,
            rating=detail_res.get("vote_average", 0.0),
            genres=[g["name"] for g in detail_res.get("genres", [])],
            is_adult=detail_res.get("adult", False),
            synopsis=detail_res.get("overview", ""),
            runtime=detail_res.get("runtime", 0),
            trailer_url=self._extract_trailer_url(detail_res),
            cast=self._extract_cast_and_crew(detail_res),
            movie_providers=self._extract_providers(detail_res),
            keywords=self._extract_keywords(detail_res),
            collection=self._extract_collection(detail_res),
            production_companies=self._extract_production_companies(detail_res),
            genre_tmdb_ids=[g["id"] for g in detail_res.get("genres", [])],
        )

    def _extract_trailer_url(self, detail_res: dict) -> str | None:
        videos = detail_res.get("videos", {}).get("results", [])
        trailer_key = next(
            (
                v["key"]
                for v in videos
                if v["type"] == "Trailer" and v["site"] == "YouTube"
            ),
            None,
        )
        return f"https://www.youtube.com/watch?v={trailer_key}" if trailer_key else None

    def _extract_providers(self, detail_res: dict) -> list[MovieProvider]:
        prov_data = (
            detail_res.get("watch/providers", {}).get("results", {}).get("US", {})
        )
        providers = []
        for p_type in ["flatrate", "rent", "buy"]:
            for item in prov_data.get(p_type, []):
                providers.append(
                    MovieProvider(name=item["provider_name"], provider_type=p_type)
                )
        return providers

    def _extract_cast_and_crew(self, detail_res: dict) -> list[CastMember]:
        credits = detail_res.get("credits", {})
        members: list[CastMember] = []

        for c in credits.get("cast", [])[:5]:
            members.append(
                CastMember(
                    name=c["name"],
                    role_type="Actor",
                    character_name=c.get("character"),
                    profile_path=f"{self.IMG_URL}{c['profile_path']}"
                    if c.get("profile_path")
                    else None,
                    tmdb_person_id=c.get("id"),
                )
            )

        producer_count = 0
        writer_count = 0
        for c in credits.get("crew", []):
            job = c.get("job", "")
            profile = (
                f"{self.IMG_URL}{c['profile_path']}" if c.get("profile_path") else None
            )
            person_id = c.get("id")

            if job == "Director":
                members.append(
                    CastMember(
                        name=c["name"],
                        role_type="Director",
                        character_name=None,
                        profile_path=profile,
                        tmdb_person_id=person_id,
                    )
                )
            elif job == "Producer" and producer_count < 3:
                members.append(
                    CastMember(
                        name=c["name"],
                        role_type="Producer",
                        character_name=None,
                        profile_path=profile,
                        tmdb_person_id=person_id,
                    )
                )
                producer_count += 1
            elif job in ("Screenplay", "Writer") and writer_count < 3:
                members.append(
                    CastMember(
                        name=c["name"],
                        role_type="Writer",
                        character_name=None,
                        profile_path=profile,
                        tmdb_person_id=person_id,
                    )
                )
                writer_count += 1

        return members

    def _extract_keywords(self, detail_res: dict) -> list[TMDBKeyword]:
        keywords_data = detail_res.get("keywords", {}).get("keywords", [])
        return [TMDBKeyword(tmdb_id=kw["id"], name=kw["name"]) for kw in keywords_data]

    def _extract_collection(self, detail_res: dict) -> TMDBCollection | None:
        coll = detail_res.get("belongs_to_collection")
        if not coll:
            return None
        return TMDBCollection(tmdb_id=coll["id"], name=coll["name"])

    def _extract_production_companies(
        self, detail_res: dict
    ) -> list[TMDBProductionCompany]:
        companies = detail_res.get("production_companies", [])
        return [
            TMDBProductionCompany(
                tmdb_id=pc["id"],
                name=pc["name"],
                origin_country=pc.get("origin_country"),
            )
            for pc in companies
        ]


class MovieHydrator:
    def __init__(self, db_session_factory, neo4j_driver=None) -> None:
        self.db_session_factory = db_session_factory
        self.neo4j_driver = neo4j_driver
        self.tmdb = TMDBFetcher()
        self._kg_semaphore: asyncio.Semaphore | None = None

    def _get_kg_semaphore(self) -> asyncio.Semaphore:
        # Lazy init — asyncio.Semaphore must be created inside a running loop
        if self._kg_semaphore is None:
            self._kg_semaphore = asyncio.Semaphore(3)
        return self._kg_semaphore

    async def get_or_fetch_movie(
        self, movie_db_id: int, movie_title: str
    ) -> MovieDetails | None:
        """Get movie from DB by internal ID, or search TMDB by title and persist."""
        logger.info(f"Hydrating movie: id={movie_db_id}, title={movie_title}")

        async with self.db_session_factory() as db:
            movie = await get_movie_by_id(db, movie_db_id)
            if movie and movie.tmdb_id:
                logger.info(f"Movie found in DB: {movie.title}")
                return await movie_to_details(db, movie_db_id)

        detail_res = await self.tmdb.fetch_detail_by_title(movie_title)
        if not detail_res:
            logger.warning(f"Could not find movie on TMDB: {movie_title}")
            return None

        details = self.tmdb.parse_detail_response(movie_db_id, detail_res)
        logger.info(f"Fetched movie from TMDB: {details.title}")

        async with self.db_session_factory() as db:
            try:
                await save_hydrated_movie(db, movie_db_id, details)
            except Exception:
                logger.debug(
                    f"Movie {movie_db_id} already saved by another task, skipping"
                )

        if self.neo4j_driver:
            asyncio.create_task(self._enrich_kg(details))

        return details

    async def _enrich_kg(self, movie_details: MovieDetails) -> None:
        """Fire-and-forget KG enrichment. Never blocks the feed.

        Semaphore (max 3 concurrent) prevents Neo4j deadlocks that occur when
        many movies concurrently MERGE the same shared nodes (actors, genres).
        """
        async with self._get_kg_semaphore():
            try:
                from movie_recommender.services.knowledge_graph.writer import (
                    upsert_movie_to_kg,
                )

                await upsert_movie_to_kg(self.neo4j_driver, movie_details)
            except Exception:
                logger.warning(
                    "KG enrichment failed for %s, continuing", movie_details.title
                )
