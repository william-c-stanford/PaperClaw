"""Image generation for paper figures — OpenAI-style images API.

Calls a configurable `/images/generations` endpoint (`LLMSettings.image_*`) to
render an introduction / methodology figure from a text prompt and save it as a
PNG. Provider-agnostic over the OpenAI image API shape (gpt-image-1 returns
`b64_json`; DALL·E returns a `url`). Disabled (returns False) when no image key
is configured, so callers fall back to a matplotlib diagram.
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx

from paperclaw.config import LLMSettings


def is_configured(settings: LLMSettings) -> bool:
    return bool(getattr(settings, "image_api_key", ""))


def generate_image(settings: LLMSettings, prompt: str, out_path: Path) -> bool:
    """Generate one image for *prompt* and write it to *out_path* (PNG). Returns
    True on success, False if not configured or on any error (best-effort)."""
    key = getattr(settings, "image_api_key", "") or ""
    if not key:
        return False
    base = (getattr(settings, "image_base_url", None) or "https://api.openai.com/v1").rstrip("/")
    model = getattr(settings, "image_model", None) or "gpt-image-1"
    try:
        r = httpx.post(
            f"{base}/images/generations",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "prompt": prompt, "n": 1, "size": "1024x1024"},
            timeout=180,
        )
        r.raise_for_status()
        data = (r.json().get("data") or [{}])[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if data.get("b64_json"):
            out_path.write_bytes(base64.b64decode(data["b64_json"]))
            return True
        if data.get("url"):
            img = httpx.get(data["url"], timeout=180)
            img.raise_for_status()
            out_path.write_bytes(img.content)
            return True
        return False
    except Exception:
        return False
