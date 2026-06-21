"""OpenAlex client for searching recent, highly-cited papers.

OpenAlex (https://openalex.org) is a completely open academic graph with no
authentication required and generous rate limits (100k req/day in the polite
pool).

Searches match on TITLE + ABSTRACT only (`title_and_abstract.search`), not
full text — full-text matching ranks generic surveys above the actually
on-topic papers for a specific query. This mirrors OpenAlex's web UI.

Two search strategies for domain generation:
  search_recent_papers  — last 5 years, relevance-sorted, cited_by_count>20
                          (finds well-established relevant papers)
  search_sota_papers    — last 2 calendar years, relevance-sorted, no citation
                          floor, INCLUDING preprints (the newest work — arXiv /
                          SSRN diffusion-style SOTA — is typed `preprint` in
                          OpenAlex and would be dropped by a `type:article` filter)
"""

import asyncio
import os
from datetime import date

import httpx

_OPENALEX_WORKS = "https://api.openalex.org/works"
_TIMEOUT = 12.0
_HEADERS = {"User-Agent": "PaperClaw"}

# OpenAlex moved to a credit/budget model: anonymous requests get a tiny daily
# budget per IP (HTTP 429 "Insufficient budget" once spent, resets midnight UTC).
# Configure an OPENALEX_API_KEY for an authenticated budget — it surfaces in Settings
# and is sent on every request when present. `configure()` is called from
# config.load_settings so every entry point (server / CLI / detached children) picks
# up the saved key; an env var still wins (load_settings merges it in). The last fetch
# error (e.g. a rate-limit reason) is recorded so callers show an honest status
# instead of a misleading "found 0 papers".
_last_error: str | None = None
_creds: dict[str, str] = {"api_key": ""}


def configure(api_key: str | None = None) -> None:
    """Set the OpenAlex API key used on every request (from settings/env)."""
    _creds["api_key"] = (api_key or "").strip()


def _auth_params() -> dict:
    # configured value (settings, already env-merged by load_settings) first, then a
    # raw env var as a fallback for code paths that import us before settings load.
    key = _creds["api_key"] or os.environ.get("OPENALEX_API_KEY", "").strip()
    return {"api_key": key} if key else {}


def last_error() -> str | None:
    """The reason the most recent OpenAlex fetch returned nothing (rate-limit /
    network / HTTP error), or None on success. Lets the UI/CLI explain an empty
    result instead of silently reporting 'found 0 papers'."""
    return _last_error


def _note_status(resp: "httpx.Response") -> None:
    """Record a non-200 OpenAlex response as a human-readable error reason."""
    global _last_error
    if resp.status_code == 429:
        _last_error = ("OpenAlex rate-limited (HTTP 429 — daily budget exhausted; "
                       "resets midnight UTC). Set OPENALEX_API_KEY for a dedicated budget.")
    else:
        _last_error = f"OpenAlex HTTP {resp.status_code}"

_SELECT = ",".join([
    "title",
    "authorships",
    "publication_year",
    "primary_location",
    "cited_by_count",
    "abstract_inverted_index",
    "doi",
])

_BROAD_LOOKBACK_YEARS = 5
_BROAD_MIN_CITATIONS = 20   # modest floor: title+abstract search is already on-topic,
                            # and a high floor empties out niche/recent subfields


def _clean_query(query: str) -> str:
    """Strip characters that have meaning inside an OpenAlex filter value
    (commas separate filters; pipes mean OR) so a domain prompt can't break
    the `title_and_abstract.search:<query>` filter."""
    return query.replace(",", " ").replace("|", " ").strip()


def _rebuild_abstract(inverted: dict | None) -> str:
    """Reconstruct plain text from OpenAlex's inverted-index abstract format."""
    if not inverted:
        return ""
    words: dict[int, str] = {}
    for word, positions in inverted.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[i] for i in sorted(words))


def _parse_works(raw_results: list[dict], limit: int) -> list[dict]:
    papers = []
    for work in raw_results:
        title = (work.get("title") or "").strip()
        if not title:
            continue
        authors = [
            a["author"]["display_name"]
            for a in (work.get("authorships") or [])[:4]
            if a.get("author", {}).get("display_name")
        ]
        loc = work.get("primary_location") or {}
        src = loc.get("source") or {}
        venue = (src.get("display_name") or "").strip()
        abstract_raw = _rebuild_abstract(work.get("abstract_inverted_index"))
        papers.append({
            "title": title,
            "authors": authors,
            "year": work.get("publication_year"),
            "venue": venue,
            "citations": work.get("cited_by_count") or 0,
            "abstract": abstract_raw[:300].strip(),
            "doi": (work.get("doi") or "").replace("https://doi.org/", ""),
        })
        if len(papers) >= limit:
            break
    return papers


async def _fetch(params: dict) -> list[dict]:
    """GET /works with given params; returns raw results list or [] on failure
    (recording the reason in :func:`last_error`)."""
    global _last_error
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as client:
            resp = await client.get(_OPENALEX_WORKS, params={**params, **_auth_params()})
    except Exception as exc:  # network / timeout
        _last_error = f"OpenAlex request failed: {type(exc).__name__}"
        return []
    if resp.status_code != 200:
        _note_status(resp)
        return []
    _last_error = None
    return resp.json().get("results", [])


def _fetch_sync(params: dict) -> list[dict]:
    """Blocking GET /works — for the tool-call executor, which runs synchronously
    inside an already-running event loop (so the async client can't be awaited)."""
    global _last_error
    try:
        with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS) as client:
            resp = client.get(_OPENALEX_WORKS, params={**params, **_auth_params()})
    except Exception as exc:
        _last_error = f"OpenAlex request failed: {type(exc).__name__}"
        return []
    if resp.status_code != 200:
        _note_status(resp)
        return []
    _last_error = None
    return resp.json().get("results", [])


def search_papers_sync(query: str, limit: int = 8, recent_only: bool = False) -> list[dict]:
    """Synchronous OpenAlex search used by the ``openalex_search`` LLM tool.

    ``recent_only=False`` mirrors :func:`search_recent_papers` (last 5 years,
    citation floor, established work); ``recent_only=True`` mirrors
    :func:`search_sota_papers` (last 2 calendar years, no floor, includes
    preprints). Returns parsed paper dicts (same shape as the async helpers).
    """
    q = _clean_query(query)
    if not q:
        return []
    today = date.today()
    if recent_only:
        since = today.year - 1
        filt = (
            f"title_and_abstract.search:{q},type:article|preprint,"
            f"publication_year:{since}-{today.year}"
        )
    else:
        since = today.year - _BROAD_LOOKBACK_YEARS
        filt = (
            f"title_and_abstract.search:{q},type:article,"
            f"cited_by_count:>{_BROAD_MIN_CITATIONS},"
            f"publication_year:{since}-{today.year}"
        )
    raw = _fetch_sync({
        "filter": filt,
        "per-page": min(limit * 2, 25),
        "select": _SELECT,
    })
    return _parse_works(raw, limit)


async def search_recent_papers(query: str, limit: int = 8) -> list[dict]:
    """Relevant, established papers from the last 5 years.

    Matches title + abstract only (relevance-sorted by default) with a modest
    citation floor so results stay on-topic and field-appropriate.
    """
    today = date.today()
    since = today.year - _BROAD_LOOKBACK_YEARS
    q = _clean_query(query)
    raw = await _fetch({
        # no 'sort' → OpenAlex defaults to relevance_score:desc
        "filter": (
            f"title_and_abstract.search:{q},type:article,"
            f"cited_by_count:>{_BROAD_MIN_CITATIONS},publication_year:{since}-{today.year}"
        ),
        "per-page": min(limit * 2, 25),
        "select": _SELECT,
    })
    return _parse_works(raw, limit)


async def search_sota_papers(query: str, limit: int = 6) -> list[dict]:
    """Most relevant papers from the last 2 *calendar* years (current + previous).

    No citation floor — new papers haven't had time to accumulate citations.
    Includes preprints (`type:article|preprint`): the newest SOTA work lives on
    arXiv / SSRN and is typed `preprint`, so excluding it would drop exactly the
    cutting-edge papers this section is meant to surface.
    Year window: (today.year - 1) to today.year, e.g. 2025-2026.
    """
    today = date.today()
    since = today.year - 1   # previous calendar year, e.g. 2025 when today is 2026
    q = _clean_query(query)
    raw = await _fetch({
        "filter": (
            f"title_and_abstract.search:{q},type:article|preprint,"
            f"publication_year:{since}-{today.year}"
        ),
        "per-page": min(limit * 2, 25),
        "select": _SELECT,
    })
    return _parse_works(raw, limit)


async def search_for_domain(query: str) -> tuple[list[dict], list[dict]]:
    """Run both searches concurrently; deduplicate SOTA against broad by title."""
    broad, sota = await asyncio.gather(
        search_recent_papers(query, limit=8),
        search_sota_papers(query, limit=6),
    )
    broad_titles = {p["title"].lower() for p in broad}
    sota = [p for p in sota if p["title"].lower() not in broad_titles]
    return broad, sota


def _fmt_paper(p: dict) -> str:
    authors = p["authors"]
    if len(authors) > 2:
        auth_str = authors[0] + " et al."
    elif authors:
        auth_str = " & ".join(authors)
    else:
        auth_str = "Unknown"
    year = f" ({p['year']})" if p["year"] else ""
    venue = f" — {p['venue']}" if p["venue"] else ""
    cites = f" [{p['citations']:,} citations]" if p.get("citations") else ""
    line = f"- {auth_str}{year}. {p['title']}{venue}{cites}"
    if p["abstract"]:
        line += f"\n  {p['abstract']}…"
    return line


def paper_label(p: dict) -> str:
    """Compact one-line label for UI/CLI display: 'Author et al. (Year). Title'."""
    authors = p.get("authors") or []
    if len(authors) > 2:
        auth = authors[0] + " et al."
    elif authors:
        auth = " & ".join(authors)
    else:
        auth = "Unknown"
    year = f" ({p['year']})" if p.get("year") else ""
    return f"{auth}{year}. {p.get('title', '')}"


def format_papers_for_prompt(papers: list[dict]) -> str:
    """Format a flat list (used by brainstorm / fallback paths)."""
    if not papers:
        return ""
    lines = ["=== Relevant highly-cited papers (OpenAlex) ==="]
    lines += [_fmt_paper(p) for p in papers]
    lines.append("=== end ===")
    return "\n".join(lines)


def format_domain_papers_for_prompt(broad: list[dict], sota: list[dict]) -> str:
    """Format two paper lists (established context + SOTA) for domain-spec generation."""
    if not broad and not sota:
        return ""
    today = date.today()
    sota_since = today.year - 1
    lines: list[str] = []
    if broad:
        lines.append(f"=== Established relevant papers — last {_BROAD_LOOKBACK_YEARS} years (OpenAlex, relevance-sorted) ===")
        lines += [_fmt_paper(p) for p in broad]
        lines.append("")
    if sota:
        lines.append(f"=== SOTA: most recent papers {sota_since}–{today.year} (OpenAlex, relevance-sorted) ===")
        lines += [_fmt_paper(p) for p in sota]
    lines.append("=== end of OpenAlex search results ===")
    return "\n".join(lines)
