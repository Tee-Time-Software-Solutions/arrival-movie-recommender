from fastapi import APIRouter


router = APIRouter(prefix="/interactions")


@router.post(path="/{movie_id}/swipe")
async def register_movie_feedback() -> RegisteredFeedback:
    """
    1. Register in db for a given movie like, dislike and if supercharged
    2. Trigger recommder update based on this info
    3. If movie not in db return 404 error

    Dependencies:
          - Extract current user
    """
