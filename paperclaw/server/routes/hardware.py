"""Hardware / environment detection routes.

GET  /api/hardware         — persisted snapshot + SSH config + HARDWARE.md
POST /api/hardware/detect  — probe local + remotes now, write HARDWARE.md
PUT  /api/hardware/ssh     — save the SSH target list (no re-detection)
"""

from fastapi import APIRouter, Request

from paperclaw import service
from paperclaw.server.models import HardwareView, RunConfig, SSHTargetsUpdate

router = APIRouter(prefix="/api/hardware", tags=["hardware"])


@router.get("", response_model=HardwareView)
def get_hardware(request: Request):
    return service.get_hardware_view(request.app.state.store)


@router.post("/detect", response_model=HardwareView)
async def detect_hardware(request: Request):
    return await service.detect_hardware(
        request.app.state.store, request.app.state.settings
    )


@router.put("/ssh", response_model=HardwareView)
def put_ssh_targets(body: SSHTargetsUpdate, request: Request):
    return service.save_ssh_targets(request.app.state.store, body.ssh_targets)


@router.put("/run-config", response_model=HardwareView)
def put_run_config(body: RunConfig, request: Request):
    return service.save_run_config(request.app.state.store, body)
