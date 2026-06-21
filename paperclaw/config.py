"""Application home directory and persisted settings."""

import json
import os
from pathlib import Path

from pydantic import BaseModel

DEFAULT_MODEL = "claude-opus-4-8"


def paperclaw_home() -> Path:
    """Workspace root: $PAPERCLAW_HOME, else ``./saves`` (relative to the working dir).

    Always returned ABSOLUTE — so detached children we spawn (experiment jobs,
    `auto` runs) that inherit PAPERCLAW_HOME and run with a different cwd resolve to the
    SAME directory instead of doubling a relative path (e.g. ``saves/saves``)."""
    return Path(os.environ.get("PAPERCLAW_HOME", "saves")).resolve()


def claude_cli_available() -> bool:
    """True if the `claude` (Claude Code) CLI is on PATH — the recommended headless
    coding agent for experiments (`cli` experiment mode)."""
    import shutil
    return shutil.which("claude") is not None


class LLMSettings(BaseModel):
    provider: str = "anthropic"  # "anthropic" | "openai" (OpenAI-compatible)
    base_url: str | None = None
    api_key: str = ""
    model: str = DEFAULT_MODEL
    # Chat file editor: "deepagents" (default — used when the optional package is
    # installed; otherwise falls back to "builtin" automatically) | "builtin".
    chat_agent: str = "deepagents"
    # Image generation API for paper figures (intro/method diagrams). OpenAI-style
    # images endpoint by default; base_url None = OpenAI. Empty key = disabled.
    image_base_url: str | None = None
    image_api_key: str = ""
    image_model: str | None = None  # e.g. "gpt-image-1", "dall-e-3"
    # OpenAlex literature search (domain survey, SOTA, references). OpenAlex now
    # budget-limits anonymous (per-IP) requests; an api_key gets a dedicated budget.
    # Empty = anonymous (may hit HTTP 429).
    openalex_api_key: str = ""


def settings_path(home: Path) -> Path:
    return home / "settings.json"


def _parse_env_file(path: Path) -> dict[str, str]:
    """Minimal .env parser: KEY=VALUE lines, # comments, optional quotes."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip("'\"")
    return out


def load_settings(home: Path) -> LLMSettings:
    """Settings precedence: env vars > .env (cwd, then PaperClaw home) > settings.json.

    Recognized keys: PAPERCLAW_PROVIDER, PAPERCLAW_BASE_URL, PAPERCLAW_MODEL, PAPERCLAW_API_KEY —
    plus ANTHROPIC_API_KEY / OPENAI_API_KEY as provider-matched fallbacks.
    """
    settings = LLMSettings()
    path = settings_path(home)
    if path.is_file():
        try:
            settings = LLMSettings.model_validate_json(path.read_text())
        except Exception:
            pass

    env: dict[str, str] = {}
    env.update(_parse_env_file(home / ".env"))
    env.update(_parse_env_file(Path.cwd() / ".env"))
    env.update({k: v for k, v in os.environ.items() if v})

    if env.get("PAPERCLAW_PROVIDER") in ("anthropic", "openai"):
        settings.provider = env["PAPERCLAW_PROVIDER"]
    if env.get("PAPERCLAW_BASE_URL"):
        settings.base_url = env["PAPERCLAW_BASE_URL"]
    if env.get("PAPERCLAW_MODEL"):
        settings.model = env["PAPERCLAW_MODEL"]
    if env.get("PAPERCLAW_CHAT_AGENT"):
        settings.chat_agent = env["PAPERCLAW_CHAT_AGENT"]
    if env.get("PAPERCLAW_IMAGE_BASE_URL"):
        settings.image_base_url = env["PAPERCLAW_IMAGE_BASE_URL"]
    if env.get("PAPERCLAW_IMAGE_API_KEY"):
        settings.image_api_key = env["PAPERCLAW_IMAGE_API_KEY"]
    if env.get("PAPERCLAW_IMAGE_MODEL"):
        settings.image_model = env["PAPERCLAW_IMAGE_MODEL"]
    if env.get("PAPERCLAW_API_KEY"):
        settings.api_key = env["PAPERCLAW_API_KEY"]
    elif not settings.api_key:
        fallback = "ANTHROPIC_API_KEY" if settings.provider == "anthropic" else "OPENAI_API_KEY"
        if env.get(fallback):
            settings.api_key = env[fallback]
    # OpenAlex key: env (or .env) overrides the saved setting, like the LLM key.
    if env.get("OPENALEX_API_KEY"):
        settings.openalex_api_key = env["OPENALEX_API_KEY"]

    # Push the resolved OpenAlex key into the literature client so every entry
    # point (server / CLI / detached children all call load_settings) uses it.
    from paperclaw import literature
    literature.configure(settings.openalex_api_key)
    return settings


def save_settings(home: Path, settings: LLMSettings) -> None:
    home.mkdir(parents=True, exist_ok=True)
    path = settings_path(home)
    path.write_text(json.dumps(settings.model_dump(), indent=2))
    os.chmod(path, 0o600)  # contains the API key
