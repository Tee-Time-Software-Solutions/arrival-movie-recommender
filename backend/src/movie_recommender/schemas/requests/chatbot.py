from pydantic import BaseModel


class ChatMessagePayload(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessagePayload] | None = None
