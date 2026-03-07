import logging

from movie_recommender.core.settings.main import AppSettings
from sqlalchemy.ext.asyncio import AsyncSession
import requests

from movie_recommender.database.CRUD.movies import (
    get_movie_by_id,
    save_hydrated_movie,
    movie_to_details,
)
from movie_recommender.schemas.requests.movies import (
    CastMember,
    MovieDetails,
    MovieProvider,
)

logger = logging.getLogger(__name__)


class MovieHydrator:
    def __init__(self, db_session: AsyncSession) -> None:
        self.db = db_session
        self.settings = AppSettings()
        self.TMDB_API_KEY = self.settings.tmdb.api_key
        self.BASE_URL = self.settings.tmdb.base_url
        self.IMG_URL = self.settings.tmdb.img_url

    async def get_or_fetch_movie(self, movie_db_id: int, movie_title: str):
        """
        Get movie from DB or fetch from TMDB and store if not found.
        movie_db_id is the PK in our DB, originally created by the recommender service.
        """
        logger.info(f"Hydrating movie: id={movie_db_id}, title={movie_title}")

        movie = await get_movie_by_id(self.db, movie_db_id)
        if movie and movie.tmdb_id:
            logger.info(f"Movie found in DB: {movie.title}")
            return movie_to_details(movie)

        movie_details: MovieDetails = self._fetch_tmdb_metadata(
            movie_db_id, movie_title
        )
        if not movie_details:
            logger.warning(f"Could not find movie on TMDB: {movie_title}")
            return None

        logger.info(f"Fetched movie from TMDB: {movie_details.title}")
        await save_hydrated_movie(self.db, movie_db_id, movie_details)
        return movie_details

    def _fetch_tmdb_metadata(
        self, movie_db_id: int, movie_title: str
    ) -> MovieDetails | None:
        logger.info(f"Searching TMDB for: {movie_title}")
        search_res = requests.get(
            f"{self.BASE_URL}/search/movie",
            params={"api_key": self.TMDB_API_KEY, "query": movie_title},
        ).json()
        if not search_res["results"]:
            return None

        tmdb_movie = search_res["results"][0]
        tmdb_id = tmdb_movie["id"]

        detail_res = requests.get(
            f"{self.BASE_URL}/movie/{tmdb_id}",
            params={
                "api_key": self.TMDB_API_KEY,
                "append_to_response": "credits,videos,watch/providers",
            },
        ).json()

        videos = detail_res.get("videos", {}).get("results", [])
        trailer_key = next(
            (
                v["key"]
                for v in videos
                if v["type"] == "Trailer" and v["site"] == "YouTube"
            ),
            None,
        )
        trailer_url = (
            f"https://www.youtube.com/watch?v={trailer_key}" if trailer_key else None
        )

        prov_data = (
            detail_res.get("watch/providers", {}).get("results", {}).get("US", {})
        )
        providers = []
        for p_type in ["flatrate", "rent", "buy"]:
            if p_type in prov_data:
                for item in prov_data[p_type]:
                    providers.append(
                        MovieProvider(name=item["provider_name"], provider_type=p_type)
                    )

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
            cast=[
                CastMember(
                    name=c["name"],
                    role_type=c.get("character"),
                    profile_path=f"{self.IMG_URL}{c['profile_path']}"
                    if c.get("profile_path")
                    else None,
                )
                for c in detail_res.get("credits", {}).get("cast", [])[:5]
            ],
            movie_providers=providers,
        )
