"""fetch_url tool — fetch a web page and return its readable text.

The natural companion to ``web_search``: read a result (or any URL) so the agent
can ground its answer in the page content. Keyless httpx GET; HTML is reduced to
plain text and truncated. http(s) only.
"""

from __future__ import annotations

import html as _html
import re
from pathlib import Path
from typing import Any

import httpx

SCHEMA: dict[str, Any] = {
    "name": "fetch_url",
    "description": (
        "Fetch a web page (http/https) and return its readable text content, "
        "truncated. Use after web_search to read a result, or to read a known "
        "URL. Returns plain text only (no images/scripts)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The http(s) URL to fetch."},
        },
        "required": ["url"],
    },
}

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
}
_TIMEOUT = 15.0
_MAX_CHARS = 8000

_DROP_RE = re.compile(r"<(script|style|noscript|head)\b[^>]*>.*?</\1>", re.DOTALL | re.I)
_TAG_RE = re.compile(r"<[^>]+>")


def _html_to_text(page: str) -> str:
    text = _DROP_RE.sub(" ", page)
    text = _TAG_RE.sub(" ", text)
    text = _html.unescape(text)
    text = re.sub(r"[ \t\r\f]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


def _fetch(url: str) -> str:
    with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS, follow_redirects=True) as c:
        resp = c.get(url)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    if "html" in ctype or "<html" in resp.text[:500].lower():
        return _html_to_text(resp.text)
    return resp.text


def execute(base_dir: Path, inputs: dict[str, Any]) -> str:
    """Tool entry point. ``base_dir`` is unused (fetch is workspace-agnostic)."""
    url = (inputs.get("url") or "").strip()
    if not url:
        return "Error: 'url' is required."
    if not url.startswith(("http://", "https://")):
        return "Error: only http/https URLs are supported."
    try:
        text = _fetch(url)
    except Exception as exc:  # network/HTTP errors must not break the tool loop
        return f"Error fetching {url}: {exc}"
    if not text:
        return f"Fetched {url} but found no readable text."
    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + f"\n…[truncated at {_MAX_CHARS} chars]"
    return f"Content of {url}:\n\n{text}"
