"""web_search tool — general internet search (DuckDuckGo, no API key).

Complements ``openalex_search`` (academic papers): use this for current events,
documentation, software, datasets, non-paper facts. Keyless — scrapes the
DuckDuckGo HTML endpoint with httpx (same no-auth approach as ``literature.py``).
Pair with ``fetch_url`` to read a result page.

Registered in ``tools/__init__.py``; picked up by the chat tool loop for both
providers. Runs synchronously inside the async loop (like ``openalex_search``).
"""

from __future__ import annotations

import html as _html
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

SCHEMA: dict[str, Any] = {
    "name": "web_search",
    "description": (
        "Search the public web (DuckDuckGo, no key needed) and return the top "
        "results as title / URL / snippet. Use for current events, docs, "
        "software, datasets, and other non-academic info; use openalex_search for "
        "papers. Follow up with fetch_url to read a result. NEVER fabricate "
        "results or URLs — if nothing comes back, say so."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query (a few focused keywords)."},
            "max_results": {"type": "integer", "description": "Max results to return (1-10, default 6)."},
        },
        "required": ["query"],
    },
}

_ENDPOINT = "https://html.duckduckgo.com/html/"
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
}
_TIMEOUT = 12.0
_MAX = 10

_RESULT_RE = re.compile(r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL | re.I)
_SNIPPET_RE = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL | re.I)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip(fragment: str) -> str:
    return re.sub(r"\s+", " ", _html.unescape(_TAG_RE.sub("", fragment))).strip()


def _clean_url(href: str) -> str:
    """Resolve DuckDuckGo's redirect wrapper (//duckduckgo.com/l/?uddg=...) to the real URL."""
    if href.startswith("//"):
        href = "https:" + href
    if "duckduckgo.com/l/" in href and "uddg=" in href:
        q = parse_qs(urlparse(href).query)
        if q.get("uddg"):
            return unquote(q["uddg"][0])
    return href


def _parse_results(page: str, limit: int) -> list[dict]:
    titles = _RESULT_RE.findall(page)
    snippets = _SNIPPET_RE.findall(page)
    out: list[dict] = []
    for i, (href, title) in enumerate(titles[:limit]):
        out.append({
            "title": _strip(title),
            "url": _clean_url(href),
            "snippet": _strip(snippets[i]) if i < len(snippets) else "",
        })
    return out


def _search(query: str, limit: int) -> list[dict]:
    with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS, follow_redirects=True) as c:
        resp = c.post(_ENDPOINT, data={"q": query})
    if resp.status_code != 200:
        return []
    return _parse_results(resp.text, limit)


def execute(base_dir: Path, inputs: dict[str, Any]) -> str:
    """Tool entry point. ``base_dir`` is unused (search is workspace-agnostic)."""
    query = (inputs.get("query") or "").strip()
    if not query:
        return "Error: 'query' is required."
    try:
        limit = max(1, min(int(inputs.get("max_results") or 6), _MAX))
    except (TypeError, ValueError):
        limit = 6
    try:
        results = _search(query, limit)
    except Exception as exc:  # network/parse failures must not break the tool loop
        return f"Error searching the web: {exc}"
    if not results:
        return (f"No web results for {query!r} (the search may be rate-limited). "
                "Do NOT fabricate results — rephrase or try again.")
    return f"Web results for {query!r}:\n" + "\n".join(
        f"- {r['title']}\n  {r['url']}\n  {r['snippet']}" for r in results
    )
