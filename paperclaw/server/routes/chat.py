import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from paperclaw import service
from paperclaw.server.models import ChatContext, Message

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    idea_id: str | None = Field(default=None, alias="ideaId")
    seed_id: str | None = Field(default=None, alias="seedId")
    domain_id: str | None = Field(default=None, alias="domainId")
    content: str

    model_config = {"populate_by_name": True}


@router.get("/contexts", response_model=list[ChatContext])
def list_contexts(request: Request):
    return request.app.state.store.list_contexts()


@router.get("/{context_id}/messages", response_model=list[Message])
def list_messages(context_id: str, request: Request):
    return request.app.state.store.list_messages(context_id)


@router.post("/send", response_model=list[Message])
async def send_message(body: ChatRequest, request: Request):
    try:
        return await service.send_chat(
            request.app.state.store,
            request.app.state.settings,
            body.content,
            idea_id=body.idea_id,
            seed_id=body.seed_id,
            domain_id=body.domain_id,
        )
    except service.NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/stream")
async def stream_message(body: ChatRequest, request: Request):
    """SSE endpoint — emits delta and done events so the UI can stream text."""
    store = request.app.state.store
    settings = request.app.state.settings

    async def event_gen():
        async for event in service.stream_chat_events(
            store, settings, body.content,
            idea_id=body.idea_id, seed_id=body.seed_id, domain_id=body.domain_id,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
