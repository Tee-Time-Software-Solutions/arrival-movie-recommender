import logging

from neo4j import AsyncDriver

from movie_recommender.schemas.requests.movies import MovieDetails

logger = logging.getLogger(__name__)


async def upsert_movie_to_kg(driver: AsyncDriver, details: MovieDetails) -> None:
    """Upsert a fully hydrated movie and all its relationships into the KG."""
    if not details.tmdb_id:
        return

    async with driver.session() as session:
        # Movie node
        await session.run(
            """
            MERGE (m:Movie {tmdb_id: $tmdb_id})
            SET m.title = $title,
                m.release_year = $release_year,
                m.rating = $rating,
                m.runtime = $runtime,
                m.is_adult = $is_adult
            """,
            tmdb_id=details.tmdb_id,
            title=details.title,
            release_year=details.release_year,
            rating=details.rating,
            runtime=details.runtime,
            is_adult=details.is_adult,
        )

        # Cast and crew — batched by role type
        directors = []
        actors = []
        writers = []
        producers = []
        for i, member in enumerate(details.cast):
            if not member.tmdb_person_id:
                continue
            person = {
                "person_id": member.tmdb_person_id,
                "name": member.name,
                "image_url": member.profile_path,
            }
            if member.role_type == "Director":
                directors.append(person)
            elif member.role_type == "Actor":
                person["character_name"] = member.character_name
                person["cast_order"] = i
                actors.append(person)
            elif member.role_type == "Writer":
                writers.append(person)
            elif member.role_type == "Producer":
                producers.append(person)

        if directors:
            await session.run(
                """
                UNWIND $people AS p
                MERGE (person:Person {tmdb_id: p.person_id})
                SET person.name = p.name, person.image_url = p.image_url
                WITH person
                MATCH (m:Movie {tmdb_id: $movie_id})
                MERGE (m)-[:DIRECTED_BY]->(person)
                """,
                people=directors,
                movie_id=details.tmdb_id,
            )

        if actors:
            await session.run(
                """
                UNWIND $people AS p
                MERGE (person:Person {tmdb_id: p.person_id})
                SET person.name = p.name, person.image_url = p.image_url
                WITH person, p
                MATCH (m:Movie {tmdb_id: $movie_id})
                MERGE (person)-[r:ACTED_IN]->(m)
                SET r.character_name = p.character_name, r.cast_order = p.cast_order
                """,
                people=actors,
                movie_id=details.tmdb_id,
            )

        if writers:
            await session.run(
                """
                UNWIND $people AS p
                MERGE (person:Person {tmdb_id: p.person_id})
                SET person.name = p.name, person.image_url = p.image_url
                WITH person
                MATCH (m:Movie {tmdb_id: $movie_id})
                MERGE (m)-[:WRITTEN_BY]->(person)
                """,
                people=writers,
                movie_id=details.tmdb_id,
            )

        if producers:
            await session.run(
                """
                UNWIND $people AS p
                MERGE (person:Person {tmdb_id: p.person_id})
                SET person.name = p.name, person.image_url = p.image_url
                WITH person
                MATCH (m:Movie {tmdb_id: $movie_id})
                MERGE (m)-[:PRODUCED_BY]->(person)
                """,
                people=producers,
                movie_id=details.tmdb_id,
            )

        # Genre relationships — batched
        if details.genre_tmdb_ids and len(details.genre_tmdb_ids) == len(
            details.genres
        ):
            genres = [
                {"genre_id": gid, "name": gname}
                for gname, gid in zip(details.genres, details.genre_tmdb_ids)
            ]
            await session.run(
                """
                UNWIND $genres AS g
                MERGE (genre:Genre {tmdb_id: g.genre_id})
                SET genre.name = g.name
                WITH genre
                MATCH (m:Movie {tmdb_id: $movie_id})
                MERGE (m)-[:HAS_GENRE]->(genre)
                """,
                genres=genres,
                movie_id=details.tmdb_id,
            )

        # Keyword relationships — batched
        if details.keywords:
            keywords = [
                {"kw_id": kw.tmdb_id, "name": kw.name} for kw in details.keywords
            ]
            await session.run(
                """
                UNWIND $keywords AS kw
                MERGE (k:Keyword {tmdb_id: kw.kw_id})
                SET k.name = kw.name
                WITH k
                MATCH (m:Movie {tmdb_id: $movie_id})
                MERGE (m)-[:HAS_KEYWORD]->(k)
                """,
                keywords=keywords,
                movie_id=details.tmdb_id,
            )

        # Collection relationship
        if details.collection:
            await session.run(
                """
                MERGE (c:Collection {tmdb_id: $coll_id})
                SET c.name = $name
                WITH c
                MATCH (m:Movie {tmdb_id: $movie_id})
                MERGE (m)-[r:BELONGS_TO]->(c)
                SET r.part_number = $part_number
                """,
                coll_id=details.collection.tmdb_id,
                name=details.collection.name,
                movie_id=details.tmdb_id,
                part_number=details.collection.part_number,
            )

        # Production company relationships — batched
        if details.production_companies:
            companies = [
                {
                    "pc_id": pc.tmdb_id,
                    "name": pc.name,
                    "country": pc.origin_country,
                }
                for pc in details.production_companies
            ]
            await session.run(
                """
                UNWIND $companies AS c
                MERGE (pc:ProductionCompany {tmdb_id: c.pc_id})
                SET pc.name = c.name, pc.origin_country = c.country
                WITH pc
                MATCH (m:Movie {tmdb_id: $movie_id})
                MERGE (m)-[:PRODUCED_BY]->(pc)
                """,
                companies=companies,
                movie_id=details.tmdb_id,
            )

    logger.info(f"Upserted movie to KG: {details.title} (tmdb_id={details.tmdb_id})")
