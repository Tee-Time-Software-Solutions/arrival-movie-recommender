import logging
import requests

from movie_recommender.core.settings.main import AppSettings
from movie_recommender.schemas.requests.movies import (
    CastMember,
    MovieDetails,
    MovieProvider,
)
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.CRUD.movies import (
    get_movie_by_id,
    save_hydrated_movie,
    movie_to_details,
)

logger = logging.getLogger(__name__)


class TMDBFetcher:
    def __init__(self) -> None:
        settings = AppSettings()
        self.TMDB_API_KEY = settings.tmdb.api_key
        self.BASE_URL = settings.tmdb.base_url
        self.IMG_URL = settings.tmdb.img_url

    async def get_or_fetch_movie(self, movie_database_id: int, movie_title: str):
        """
        Get movie from DB or fetch from TMDB and store if not found.
        """
        logger.info(f"Hydrating movie: id={movie_database_id}, title={movie_title}")
        # 1) Check db
        # movie = await self.db.movies.find_unique(where={"id": movie_id})
        # if movie:
        #       return movie

        # 2) Fetch from TMDB and store
        movie = self._fetch_tmdb_metadata(movie_database_id, movie_title)
        logger.info(f"Fetched movie from TMDB: {movie.title if movie else None}")
        # movie = await self.db.movie.create(movie)
        return movie

    def _fetch_tmdb_metadata(
        self, movie_database_id: int, movie_title: str
    ) -> MovieDetails:
        logger.info(f"Searching TMDB for: {movie_title}")
        search_res = requests.get(
            f"{self.BASE_URL}/search/movie",
            params={"api_key": self.TMDB_API_KEY, "query": movie_title},
        ).json()

        if not search_res["results"]:
            return None

        tmdb_id = search_res["results"][0]["id"]
        detail_res = requests.get(
            f"{self.BASE_URL}/movie/{tmdb_id}",
            params={
                "api_key": self.TMDB_API_KEY,
                "append_to_response": "credits,videos,watch/providers",
            },
        ).json()

        trailer_url = self._extract_trailer_url(detail_res)
        providers = self._extract_providers(detail_res)
        cast_members = self._extract_cast_and_crew(detail_res)

        poster_path = detail_res.get("poster_path")
        release_date = detail_res.get("release_date", "")

        return MovieDetails(
            movie_db_id=movie_db_id,
            tmdb_id=detail_res["id"],
            title=detail_res["original_title"],
            poster_url=f"{self.IMG_URL}{poster_path}" if poster_path else "",
            release_year=int(release_date[:4]) if release_date else 0,
            rating=detail_res.get("vote_average", 0.0),
            genres=[g["name"] for g in detail_res.get("genres", [])],
            is_adult=detail_res.get("adult", False),
            synopsis=detail_res.get("overview", ""),
            runtime=detail_res.get("runtime", 0),
            trailer_url=trailer_url,
            cast=cast_members,
            movie_providers=providers,
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
                )
            )

        producer_count = 0
        for c in credits.get("crew", []):
            job = c.get("job", "")
            if job == "Director":
                members.append(
                    CastMember(
                        name=c["name"],
                        role_type="Director",
                        character_name=None,
                        profile_path=f"{self.IMG_URL}{c['profile_path']}"
                        if c.get("profile_path")
                        else None,
                    )
                )
            elif job == "Producer" and producer_count < 3:
                members.append(
                    CastMember(
                        name=c["name"],
                        role_type="Producer",
                        character_name=None,
                        profile_path=f"{self.IMG_URL}{c['profile_path']}"
                        if c.get("profile_path")
                        else None,
                    )
                )
                producer_count += 1

        return members


class MovieHydrator:
    def __init__(self, db_session: AsyncSession) -> None:
        self.db = db_session
        self.tmdb = TMDBFetcher()

    async def get_or_fetch_movie(
        self, movie_db_id: int, movie_title: str
    ) -> MovieDetails | None:
        logger.info(f"Hydrating movie: id={movie_db_id}, title={movie_title}")

        movie = await get_movie_by_id(self.db, movie_db_id)
        if movie and movie.tmdb_id:
            logger.info(f"Movie found in DB: {movie.title}")
            return await movie_to_details(self.db, movie_db_id)

        movie_details = self.tmdb.fetch_tmdb_metadata(movie_db_id, movie_title)
        if not movie_details:
            logger.warning(f"Could not find movie on TMDB: {movie_title}")
            return None

        logger.info(f"Fetched movie from TMDB: {movie_details.title}")
        await save_hydrated_movie(self.db, movie_db_id, movie_details)
        return movie_details
