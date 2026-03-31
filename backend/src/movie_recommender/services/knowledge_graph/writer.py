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

        # Cast and crew relationships
        for member in details.cast:
            if not member.tmdb_person_id:
                continue

            if member.role_type == "Director":
                await session.run(
                    """
                    MERGE (p:Person {tmdb_id: $person_id})
                    SET p.name = $name, p.image_url = $image_url
                    WITH p
                    MATCH (m:Movie {tmdb_id: $movie_id})
                    MERGE (m)-[:DIRECTED_BY]->(p)
                    """,
                    person_id=member.tmdb_person_id,
                    name=member.name,
                    image_url=member.profile_path,
                    movie_id=details.tmdb_id,
                )
            elif member.role_type == "Actor":
                await session.run(
                    """
                    MERGE (p:Person {tmdb_id: $person_id})
                    SET p.name = $name, p.image_url = $image_url
                    WITH p
                    MATCH (m:Movie {tmdb_id: $movie_id})
                    MERGE (p)-[r:ACTED_IN]->(m)
                    SET r.character_name = $character_name, r.cast_order = $cast_order
                    """,
                    person_id=member.tmdb_person_id,
                    name=member.name,
                    image_url=member.profile_path,
                    movie_id=details.tmdb_id,
                    character_name=member.character_name,
                    cast_order=details.cast.index(member),
                )
            elif member.role_type == "Writer":
                await session.run(
                    """
                    MERGE (p:Person {tmdb_id: $person_id})
                    SET p.name = $name, p.image_url = $image_url
                    WITH p
                    MATCH (m:Movie {tmdb_id: $movie_id})
                    MERGE (m)-[:WRITTEN_BY]->(p)
                    """,
                    person_id=member.tmdb_person_id,
                    name=member.name,
                    image_url=member.profile_path,
                    movie_id=details.tmdb_id,
                )
            elif member.role_type == "Producer":
                await session.run(
                    """
                    MERGE (p:Person {tmdb_id: $person_id})
                    SET p.name = $name, p.image_url = $image_url
                    WITH p
                    MATCH (m:Movie {tmdb_id: $movie_id})
                    MERGE (m)-[:PRODUCED_BY]->(p)
                    """,
                    person_id=member.tmdb_person_id,
                    name=member.name,
                    image_url=member.profile_path,
                    movie_id=details.tmdb_id,
                )

        # Genre relationships
        if details.genre_tmdb_ids and len(details.genre_tmdb_ids) == len(
            details.genres
        ):
            for genre_name, genre_tmdb_id in zip(
                details.genres, details.genre_tmdb_ids
            ):
                await session.run(
                    """
                    MERGE (g:Genre {tmdb_id: $genre_id})
                    SET g.name = $name
                    WITH g
                    MATCH (m:Movie {tmdb_id: $movie_id})
                    MERGE (m)-[:HAS_GENRE]->(g)
                    """,
                    genre_id=genre_tmdb_id,
                    name=genre_name,
                    movie_id=details.tmdb_id,
                )

        # Keyword relationships
        for kw in details.keywords:
            await session.run(
                """
                MERGE (k:Keyword {tmdb_id: $kw_id})
                SET k.name = $name
                WITH k
                MATCH (m:Movie {tmdb_id: $movie_id})
                MERGE (m)-[:HAS_KEYWORD]->(k)
                """,
                kw_id=kw.tmdb_id,
                name=kw.name,
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

        # Production company relationships
        for pc in details.production_companies:
            await session.run(
                """
                MERGE (pc:ProductionCompany {tmdb_id: $pc_id})
                SET pc.name = $name, pc.origin_country = $country
                WITH pc
                MATCH (m:Movie {tmdb_id: $movie_id})
                MERGE (m)-[:PRODUCED_BY]->(pc)
                """,
                pc_id=pc.tmdb_id,
                name=pc.name,
                country=pc.origin_country,
                movie_id=details.tmdb_id,
            )

    logger.info(f"Upserted movie to KG: {details.title} (tmdb_id={details.tmdb_id})")
