from fastapi import APIRouter, Depends, HTTPException
from neo4j import AsyncDriver
from sqlalchemy.ext.asyncio import AsyncSession

from movie_recommender.database.CRUD.users import get_user_by_firebase_uid
from movie_recommender.dependencies.database import get_db
from movie_recommender.dependencies.firebase import verify_user
from movie_recommender.dependencies.neo4j import get_neo4j_driver
from movie_recommender.schemas.requests.users import LinkedMovie
from movie_recommender.services.knowledge_graph.beacon import get_person_linked_movies

router = APIRouter(prefix="/people")


@router.get(path="/{person_tmdb_id}/linked-movies")
async def get_linked_movies_for_person(
    person_tmdb_id: int,
    db: AsyncSession = Depends(get_db),
    neo4j_driver: AsyncDriver = Depends(get_neo4j_driver),
    auth_user=Depends(verify_user()),
) -> list[LinkedMovie]:
    """Return the current user's liked movies that this person appears in."""
    user = await get_user_by_firebase_uid(db, auth_user["uid"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    results = await get_person_linked_movies(
        neo4j_driver, db, user.id, person_tmdb_id
    )
    return [LinkedMovie(**m) for m in results]
