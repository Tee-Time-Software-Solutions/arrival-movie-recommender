import logging
import os
import requests

from movie_recommender.schemas.movies import CastMember, MovieDetails, MovieProvider

logger = logging.getLogger(__name__)


class MovieHydrator:
    def __init__(self, db_session) -> None:
        self.db = db_session
        self.TMDB_API_KEY = (
            "71b58f96fc68e45669adcd6e2b5d6922"  # os.getenv("TMDB_API_KEY")
        )
        self.BASE_URL = "https://api.themoviedb.org/3"
        self.IMG_URL = "https://image.tmdb.org/t/p/w500"

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
        # 1. Search for the movie to get the ID
        search_res = requests.get(
            f"{self.BASE_URL}/search/movie",
            params={"api_key": self.TMDB_API_KEY, "query": movie_title},
        ).json()
        if not search_res["results"]:
            return None

        movie_id = search_res["results"][0]["id"]

        # 2. Get full metadata
        detail_res = requests.get(
            f"{self.BASE_URL}/movie/{movie_id}",
            params={
                "api_key": self.TMDB_API_KEY,
                "append_to_response": "credits,videos,watch/providers",
            },
        ).json()

        # 3. Extract Trailer (Look for YouTube Trailer)
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

        # 4. Extract Providers (Focusing on US region for this example)
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

        # 5. Map to Pydantic Model
        movie_details = MovieDetails(
            movie_id=str(movie_database_id),
            tmdb_id=str(detail_res["id"]),
            title=detail_res["original_title"],
            poster_url=f"{self.IMG_URL}{detail_res['poster_path']}",
            release_year=int(detail_res["release_date"][:4]),
            rating=detail_res["vote_average"],
            genres=[g["name"] for g in detail_res["genres"]],
            is_adult=detail_res["adult"],
            synopsis=detail_res["overview"],
            runtime=detail_res["runtime"],
            trailer_url=trailer_url,
            cast=[
                CastMember(
                    name=c["name"],
                    role_type=c.get("character"),
                    profile_path=f"{self.IMG_URL}{c['profile_path']}"
                    if c["profile_path"]
                    else None,
                )
                for c in detail_res["credits"]["cast"][:5]  # Top 5 cast members
            ],
            movie_providers=providers,
        )

        return movie_details
