from fastapi import APIRouter, Request

from paperclaw.server.models import Resource

router = APIRouter(prefix="/api/resources", tags=["resources"])


@router.get("/{idea_id}", response_model=list[Resource])
def list_resources(idea_id: str, request: Request):
    return request.app.state.store.list_resources(idea_id)
