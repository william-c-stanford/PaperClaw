import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from paperclaw import llm, service
from paperclaw.server.models import Seed

router = APIRouter(prefix="/api/brainstorm", tags=["brainstorm"])


class SeedCreate(BaseModel):
    text: str


class GenerateRequest(BaseModel):
    hint: str | None = None
    ideaTypes: list[str] | None = None
    emphasis: list[str] | None = None
    count: int | None = None


@router.get("", response_model=list[Seed])
def list_seeds(request: Request):
    return request.app.state.store.list_seeds()


@router.post("", response_model=Seed, status_code=201)
def create_seed(body: SeedCreate, request: Request):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Seed text must not be empty")
    return request.app.state.store.add_seed(text)


@router.delete("/{seed_id}", status_code=204)
def delete_seed(seed_id: str, request: Request):
    if not request.app.state.store.remove_seed(seed_id):
        raise HTTPException(status_code=404, detail="Seed not found")


@router.post("/generate", response_model=list[Seed])
async def generate_seeds(body: GenerateRequest, request: Request):
    try:
        return await service.generate_seeds(
            request.app.state.store, request.app.state.settings,
            hint=body.hint, idea_types=body.ideaTypes,
            emphasis=body.emphasis, count=body.count,
        )
    except llm.LLMNotConfigured as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except llm.LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.post("/generate-stream")
async def generate_seeds_stream(body: GenerateRequest, request: Request):
    """SSE version of /generate — emits status and done events so the UI shows progress."""
    store = request.app.state.store
    settings = request.app.state.settings

    async def event_gen():
        async for event in service.stream_generate_seeds_events(
            store, settings, hint=body.hint, idea_types=body.ideaTypes,
            emphasis=body.emphasis, count=body.count,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
