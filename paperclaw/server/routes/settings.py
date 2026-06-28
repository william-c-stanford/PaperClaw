from fastapi import APIRouter, HTTPException, Request

from paperclaw import codex_cli, literature
from paperclaw.config import (
    LLM_PROVIDERS,
    normalize_model_for_provider,
    provider_auth_kind,
    provider_requires_api_key,
    save_settings,
)
from paperclaw.server.models import SettingsUpdate, SettingsView

router = APIRouter(prefix="/api/settings", tags=["settings"])


def _mask(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "•" * len(key)
    return f"{key[:4]}…{key[-4:]}"


def _view(settings) -> SettingsView:
    auth_kind = provider_auth_kind(settings.provider)
    ready = codex_cli.check_readiness(run_doctor=True) if settings.provider == "codex" else None
    requires_key = provider_requires_api_key(settings.provider)
    auth_configured = bool(settings.api_key) if requires_key else bool(ready and ready.subscription_auth_configured)
    return SettingsView(
        provider=settings.provider,
        baseUrl=settings.base_url,
        model=settings.model,
        apiKeyMasked=_mask(settings.api_key) if requires_key else "",
        hasKey=bool(settings.api_key) if requires_key else False,
        authKind=auth_kind,
        authConfigured=auth_configured,
        authMethod=ready.auth_method if ready else auth_kind,
        authDetail=ready.detail if ready else "",
        runtimeHealthy=ready.runtime_healthy if ready else None,
        runtimeDetail=ready.runtime_detail if ready else "",
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
        if body.provider not in LLM_PROVIDERS:
            raise HTTPException(status_code=422, detail="provider must be 'anthropic', 'openai', or 'codex'")
        settings.provider = body.provider
    if body.base_url is not None:
        settings.base_url = body.base_url.strip() or None
    if body.model is not None:
        settings.model = body.model.strip()
    settings.model = normalize_model_for_provider(settings.provider, settings.model)
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
