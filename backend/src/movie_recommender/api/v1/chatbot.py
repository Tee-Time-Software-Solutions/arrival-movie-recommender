"""SSE streaming endpoint for the chatbot agent."""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from movie_recommender.database.CRUD.users import get_user_by_firebase_uid
from movie_recommender.dependencies.chatbot import get_chatbot_agent_factory
from movie_recommender.dependencies.database import get_db
from movie_recommender.dependencies.firebase import verify_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chatbot", tags=["chatbot"])


class ChatMessagePayload(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessagePayload] | None = None


@router.post("/stream")
async def stream_chat(
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
    auth_user=Depends(verify_user()),
    agent_factory=Depends(get_chatbot_agent_factory),
):
    """Stream chatbot responses as Server-Sent Events."""
    user_row = await get_user_by_firebase_uid(db, auth_user["uid"])
    if user_row is None:
        raise HTTPException(status_code=404, detail="User not found")

    agent = agent_factory(user_id=user_row.id)

    # Build messages list for the agent
    messages = []
    if body.history:
        for msg in body.history:
            messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": body.message})

    async def event_generator():
        try:
            async for event in agent.astream_events(
                {"messages": messages}, version="v2"
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        yield {
                            "event": "token",
                            "data": json.dumps({"token": chunk.content}),
                        }

                elif kind == "on_tool_end":
                    tool_name = event["name"]
                    tool_output = event["data"].get("output", "")

                    if (
                        tool_name == "search_movies"
                        and tool_output != "No movies found matching those criteria."
                    ):
                        try:
                            movies_data = json.loads(tool_output)
                            yield {
                                "event": "movies",
                                "data": json.dumps({"movies": movies_data}),
                            }
                        except json.JSONDecodeError:
                            pass

                    elif tool_name == "get_taste_profile":
                        try:
                            profile_data = json.loads(tool_output)
                            yield {
                                "event": "taste_profile",
                                "data": json.dumps({"profile": profile_data}),
                            }
                        except json.JSONDecodeError:
                            pass

            yield {"event": "done", "data": "{}"}

        except Exception as e:
            logger.exception("Chatbot stream error")
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(event_generator())
