import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from paperclaw import llm, service
from paperclaw.server.models import Domain, DomainSpec

router = APIRouter(prefix="/api/domains", tags=["domains"])


class DomainCreate(BaseModel):
    name: str


class DomainAutoCreate(BaseModel):
    prompt: str


class DomainSelect(BaseModel):
    selected: bool


class SpecUpdate(BaseModel):
    content: str


class CodebaseSet(BaseModel):
    url: str


@router.get("", response_model=list[Domain])
def list_domains(request: Request):
    return request.app.state.store.list_domains()


@router.post("", response_model=Domain, status_code=201)
def create_domain(body: DomainCreate, request: Request):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Domain name must not be empty")
    return request.app.state.store.add_domain(name)


@router.post("/auto", response_model=Domain, status_code=201)
async def auto_create_domain(body: DomainAutoCreate, request: Request):
    """Auto mode: the LLM writes the full DOMAIN.md from a short prompt."""
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt must not be empty")
    try:
        return await service.auto_create_domain(
            request.app.state.store, request.app.state.settings, prompt
        )
    except llm.LLMNotConfigured as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except llm.LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.post("/auto-stream")
async def auto_create_domain_stream(body: DomainAutoCreate, request: Request):
    """SSE version of /auto — emits status and done events so the UI shows progress."""
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt must not be empty")
    store = request.app.state.store
    settings = request.app.state.settings

    async def event_gen():
        async for event in service.stream_auto_create_domain_events(store, settings, prompt):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.put("/{domain_id}/select", response_model=Domain)
def select_domain(domain_id: str, body: DomainSelect, request: Request):
    domain = request.app.state.store.set_domain_selected(domain_id, body.selected)
    if domain is None:
        raise HTTPException(status_code=404, detail="Domain not found")
    return domain


@router.delete("/{domain_id}", status_code=204)
def delete_domain(domain_id: str, request: Request):
    if not request.app.state.store.remove_domain(domain_id):
        raise HTTPException(status_code=404, detail="Domain not found")


@router.get("/{domain_id}/suggestions", response_model=list[str])
async def get_suggestions(domain_id: str, request: Request):
    try:
        return await service.domain_suggestions(
            request.app.state.store, request.app.state.settings, domain_id
        )
    except service.NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{domain_id}/spec", response_model=DomainSpec)
def get_spec(domain_id: str, request: Request):
    content = request.app.state.store.get_domain_spec(domain_id)
    if content is None:
        raise HTTPException(status_code=404, detail="Domain not found")
    return DomainSpec(domainId=domain_id, content=content)


@router.put("/{domain_id}/spec", response_model=DomainSpec)
def put_spec(domain_id: str, body: SpecUpdate, request: Request):
    if not request.app.state.store.put_domain_spec(domain_id, body.content):
        raise HTTPException(status_code=404, detail="Domain not found")
    return DomainSpec(domainId=domain_id, content=body.content)


@router.post("/{domain_id}/codebase", response_model=Domain)
def set_codebase(domain_id: str, body: CodebaseSet, request: Request):
    """Download a GitHub repo into the domain's reference codebase."""
    url = body.url.strip()
    if not url:
        raise HTTPException(status_code=422, detail="A GitHub repo URL is required")
    try:
        return service.set_domain_codebase(request.app.state.store, domain_id, url)
    except service.NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except service.codebase.CodebaseError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.delete("/{domain_id}/codebase", response_model=Domain)
def clear_codebase(domain_id: str, request: Request):
    try:
        return service.clear_domain_codebase(request.app.state.store, domain_id)
    except service.NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
