"""Experiment-job monitor route.

GET /api/experiments — every experiment job across all ideas (running + recent),
newest first. Drives the top-of-screen monitor.
"""

from fastapi import APIRouter, Request

from paperclaw import jobs
from paperclaw.server.models import ExperimentJob

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.get("", response_model=list[ExperimentJob])
def list_experiments(request: Request):
    return jobs.list_experiment_jobs(request.app.state.store)
