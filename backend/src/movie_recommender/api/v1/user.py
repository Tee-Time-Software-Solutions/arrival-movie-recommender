from fastapi import APIRouter


router = APIRouter(prefix="/user")


@router.get(path="/me/stats")
async def get_user_stats() -> UserStats:
    """
    1. Fetch from db the user stats
    """
