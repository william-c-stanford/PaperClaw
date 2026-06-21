"""cite tool — add a real paper to the idea's ref.bib and get its cite key.

Lets the agent build a verified bibliography while developing an idea: look up a
paper by DOI (Crossref) or query (OpenAlex), append a BibTeX entry to ``ref.bib``,
and return the Scholar-style cite key to use in IDEA.md / the paper. The agent
must never invent citations — if nothing is found it says so.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paperclaw import references

SCHEMA: dict[str, Any] = {
    "name": "cite",
    "description": (
        "Find a REAL paper (by DOI or search query) and append it to this idea's "
        "ref.bib as a BibTeX entry, returning its cite key. Use this to build a "
        "verified bibliography — never invent citations. DOI is preferred when "
        "known; otherwise pass a title/topic query. Cite the returned key in the "
        "spec or paper."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "doi": {"type": "string", "description": "DOI of the paper (preferred when known)."},
            "query": {"type": "string", "description": "Title or topic to look up if no DOI is known."},
        },
    },
}


def execute(base_dir: Path, inputs: dict[str, Any]) -> str:
    doi = (inputs.get("doi") or "").strip() or None
    query = (inputs.get("query") or "").strip() or None
    if not (doi or query):
        return "Error: provide 'doi' or 'query'."
    try:
        entry = references.build_entry(doi, query)
    except Exception as exc:  # network/parse failures must not break the tool loop
        return f"Error building citation: {exc}"
    if not entry:
        return "No matching paper found — try a different DOI or query. Do NOT invent a citation."

    bib_path = base_dir / "ref.bib"
    existing = bib_path.read_text(encoding="utf-8") if bib_path.is_file() else ""
    parsed = references.parse_bibtex(entry)
    key = parsed[0]["key"] if parsed else "?"
    if parsed and key in references.keys_in(existing):
        return f"Already in ref.bib: {key}"
    new = (existing.rstrip() + "\n\n" + entry.strip() + "\n") if existing.strip() else entry.strip() + "\n"
    bib_path.write_text(new, encoding="utf-8")
    return f"Added to ref.bib with key '{key}'. Cite it as \\cite{{{key}}}."
