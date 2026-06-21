"""Doctor / environment readiness route.

GET /api/doctor — check the BACKEND host's environment (PaperClaw home, LLM config,
chat agent, LaTeX toolchain, image generation). Pure + fast (no LLM calls).
"""

from fastapi import APIRouter, Request

from paperclaw import service
from paperclaw.server.models import DoctorReport

router = APIRouter(prefix="/api/doctor", tags=["doctor"])


@router.get("", response_model=DoctorReport)
def get_doctor(request: Request):
    return service.environment_report(request.app.state.settings, request.app.state.home)
