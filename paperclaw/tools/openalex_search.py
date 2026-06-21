"""openalex_search tool — find real, citable papers via OpenAlex.

Lets the agent look up actual literature before adding it to DOMAIN.md / IDEA.md
so it never has to fabricate citations or fall back to "(verify)" markers. Backed
by ``literature.search_papers_sync`` (no auth, no key required).

Provider-agnostic: registered in ``tools/__init__.py`` and picked up by both the
Anthropic and OpenAI-compatible tool-call loops in ``llm.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paperclaw import literature

SCHEMA: dict[str, Any] = {
    "name": "openalex_search",
    "description": (
        "Search OpenAlex (a free, open academic graph) for real, citable papers "
        "matching a query on title + abstract. Use this BEFORE adding any paper "
        "to DOMAIN.md or IDEA.md so citations are real, never invented. Returns "
        "title, authors, year, venue, citation count, DOI, and an abstract "
        "snippet for each result."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Topic or keyword phrase to search for. Keep it focused — a "
                    "few key terms (method + task/object of study), not a full "
                    "sentence. Commas and pipes are stripped automatically."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max papers to return (1-15, default 8).",
            },
            "recent_only": {
                "type": "boolean",
                "description": (
                    "If true, return only the newest work (last 2 calendar years, "
                    "including preprints, no citation floor) — use to find SOTA. "
                    "If false (default), return established papers from the last "
                    "5 years with a modest citation floor."
                ),
            },
        },
        "required": ["query"],
    },
}

_MAX_LIMIT = 15


def _fmt(p: dict) -> str:
    """One paper as a labelled block including DOI for accurate citation."""
    authors = p.get("authors") or []
    if len(authors) > 3:
        auth = ", ".join(authors[:3]) + " et al."
    elif authors:
        auth = ", ".join(authors)
    else:
        auth = "Unknown authors"
    year = p.get("year") or "n.d."
    venue = p.get("venue") or "—"
    cites = p.get("citations") or 0
    doi = p.get("doi") or "—"
    lines = [
        f"- {p.get('title', '(untitled)')}",
        f"  Authors: {auth} ({year})",
        f"  Venue: {venue} | Citations: {cites:,} | DOI: {doi}",
    ]
    if p.get("abstract"):
        lines.append(f"  Abstract: {p['abstract']}…")
    return "\n".join(lines)


def execute(base_dir: Path, inputs: dict[str, Any]) -> str:
    """Tool-call entry point. ``base_dir`` is unused (search is workspace-agnostic)."""
    query = (inputs.get("query") or "").strip()
    if not query:
        return "Error: 'query' is required."

    try:
        limit = int(inputs.get("limit") or 8)
    except (TypeError, ValueError):
        limit = 8
    limit = max(1, min(limit, _MAX_LIMIT))
    recent_only = bool(inputs.get("recent_only", False))

    try:
        papers = literature.search_papers_sync(query, limit=limit, recent_only=recent_only)
    except Exception as exc:  # network/parse failures must not break the tool loop
        return f"Error searching OpenAlex: {exc}"

    if not papers:
        return (
            f"No OpenAlex results for {query!r}. Try a broader or differently "
            "phrased query — do NOT invent citations."
        )

    scope = "recent (last 2 years, incl. preprints)" if recent_only else "established (last 5 years)"
    header = f"OpenAlex results for {query!r} — {scope}, {len(papers)} paper(s):"
    return header + "\n" + "\n".join(_fmt(p) for p in papers)
