"""Writing-style library routes.

GET  /api/writing-styles[?domainId=]            — list global (+ domain) style guides
GET  /api/writing-styles/{name}[?domainId=]     — one guide's markdown
POST /api/writing-styles                        — create/overwrite a guide
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from paperclaw import service
from paperclaw.server.models import WritingStyle

router = APIRouter(prefix="/api/writing-styles", tags=["writing-styles"])


class StyleSave(BaseModel):
    name: str
    content: str
    domainId: str | None = None


@router.get("", response_model=list[WritingStyle])
def list_styles(request: Request, domainId: str | None = None):
    return service.list_writing_styles(request.app.state.store, domainId)


@router.get("/{name}")
def get_style(name: str, request: Request, domainId: str | None = None):
    md = service.get_writing_style(request.app.state.store, domainId, name)
    if md is None:
        raise HTTPException(status_code=404, detail="Writing style not found")
    return {"name": name, "content": md}


@router.post("")
def save_style(body: StyleSave, request: Request):
    saved = service.save_writing_style(
        request.app.state.store, body.name, body.content, body.domainId)
    if saved is None:
        raise HTTPException(status_code=422, detail="Invalid style name")
    return {"name": saved}
