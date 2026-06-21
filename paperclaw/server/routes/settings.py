from fastapi import APIRouter, HTTPException, Request

from paperclaw import literature
from paperclaw.config import save_settings
from paperclaw.server.models import SettingsUpdate, SettingsView

router = APIRouter(prefix="/api/settings", tags=["settings"])


def _mask(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "•" * len(key)
    return f"{key[:4]}…{key[-4:]}"


def _view(settings) -> SettingsView:
    return SettingsView(
        provider=settings.provider,
        baseUrl=settings.base_url,
        model=settings.model,
        apiKeyMasked=_mask(settings.api_key),
        hasKey=bool(settings.api_key),
        imageBaseUrl=settings.image_base_url,
        imageModel=settings.image_model,
        imageKeyMasked=_mask(settings.image_api_key),
        hasImageKey=bool(settings.image_api_key),
        openalexKeyMasked=_mask(settings.openalex_api_key),
        hasOpenalexKey=bool(settings.openalex_api_key),
    )


@router.get("", response_model=SettingsView)
def get_settings(request: Request):
    return _view(request.app.state.settings)


@router.put("", response_model=SettingsView)
def put_settings(body: SettingsUpdate, request: Request):
    settings = request.app.state.settings
    if body.provider is not None:
        if body.provider not in ("anthropic", "openai"):
            raise HTTPException(status_code=422, detail="provider must be 'anthropic' or 'openai'")
        settings.provider = body.provider
    if body.base_url is not None:
        settings.base_url = body.base_url.strip() or None
    if body.model is not None and body.model.strip():
        settings.model = body.model.strip()
    if body.api_key is not None and body.api_key.strip():
        # empty/missing apiKey in the payload keeps the stored key
        settings.api_key = body.api_key.strip()
    if body.image_base_url is not None:
        settings.image_base_url = body.image_base_url.strip() or None
    if body.image_model is not None:
        settings.image_model = body.image_model.strip() or None
    if body.image_api_key is not None and body.image_api_key.strip():
        settings.image_api_key = body.image_api_key.strip()
    if body.openalex_api_key is not None and body.openalex_api_key.strip():
        settings.openalex_api_key = body.openalex_api_key.strip()
    save_settings(request.app.state.home, settings)
    # apply the OpenAlex key to the live literature client (no restart needed)
    literature.configure(settings.openalex_api_key)
    return _view(settings)
