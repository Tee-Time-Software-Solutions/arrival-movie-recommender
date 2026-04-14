import logging

from fastapi import APIRouter, Depends, HTTPException, status

from movie_recommender.dependencies.neo4j import get_neo4j_driver
from movie_recommender.dependencies.redis import get_async_redis

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health")


@router.get("/ping")
async def ping() -> dict:
    return {"response": "pong"}


@router.get("/dependencies")
async def check_dependencies(
    redis_client=Depends(get_async_redis),
    neo4j_driver=Depends(get_neo4j_driver),
) -> dict:
    """Readiness: settings loaded and async Redis/Neo4j connections work."""
    checks = {"status": "healthy", "checks": {}}

    # Async Redis
    try:
        await redis_client.ping()
        checks["checks"]["redis"] = "ok"
    except Exception as e:
        checks["status"] = "unhealthy"
        checks["checks"]["redis"] = f"failed: {e}"

    # Neo4j
    try:
        async with neo4j_driver.session() as session:
            await session.run("RETURN 1")
        checks["checks"]["neo4j"] = "ok"
    except Exception as e:
        checks["status"] = "unhealthy"
        checks["checks"]["neo4j"] = f"failed: {e}"

    if checks["status"] == "unhealthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=checks
        )
    return checks
