from fastapi import APIRouter

from paperclaw.server.models import Skill
from paperclaw.skills import SKILLS

router = APIRouter(prefix="/api/skills", tags=["skills"])


@router.get("", response_model=list[Skill])
def list_skills():
    return SKILLS
