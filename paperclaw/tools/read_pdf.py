"""read_pdf tool — read a compiled PDF's RENDERED text + find where the main
content ends.

`read_file` returns nothing for a binary PDF, so the agent can't tell which
rendered page the main text ends on (it would otherwise guess from source line
numbers). This reads the PDF back with pypdf and reports, per page, where the
References and Appendix begin — i.e. the main-text boundary the page limit uses —
and can dump a single page's full text. Same boundary logic as `review_paper`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SCHEMA: dict[str, Any] = {
    "name": "read_pdf",
    "description": (
        "Read a compiled PDF's RENDERED text (read_file can't — PDFs are binary). "
        "With just `path`, returns a page map and reports where the MAIN content ends: "
        "total pages, main-text page count, and the pages where References / Appendix "
        "begin (References + Appendix are excluded from the page limit). Pass `page` to "
        "dump that one page's full text. Use this to answer 'how many pages is the main "
        "text / where does it end' from the actual PDF, not the source."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the PDF, e.g. 'paper.pdf' or 'paper_v2.pdf'.",
            },
            "page": {
                "type": "integer",
                "description": "Optional 1-based page number — return that page's full text instead of the map.",
            },
        },
        "required": ["path"],
    },
}

MAX_PAGE_CHARS = 6000


def _resolve(base_dir: Path, path: str) -> Path | str:
    if not path:
        return "Error: 'path' is required."
    target = (base_dir / path).resolve()
    try:
        target.relative_to(base_dir.resolve())
    except ValueError:
        return f"Error: path escapes workspace: {path!r}"
    if not target.exists():
        return f"Error: file not found: {path}"
    if target.suffix.lower() != ".pdf":
        return f"Error: not a PDF: {path}"
    return target


def execute(base_dir: Path, inputs: dict[str, Any]) -> str:
    """Return a page map + main-content boundary, or one page's text."""
    target = _resolve(base_dir, inputs.get("path", ""))
    if isinstance(target, str):
        return target  # error
    path = inputs.get("path", "")

    from paperclaw import paper_review

    # A specific page's full text (the agent wants to read what's on it).
    page = inputs.get("page")
    if page is not None:
        pages = paper_review._pdf_pages_text(target)
        if not pages:
            return f"Could not extract text from {path} (image-only PDF or pypdf missing)."
        try:
            n = int(page)
        except (TypeError, ValueError):
            return f"Error: 'page' must be an integer, got {page!r}."
        if not 1 <= n <= len(pages):
            return f"Error: page {n} out of range (PDF has {len(pages)} pages)."
        text = pages[n - 1].strip() or "(no extractable text on this page)"
        if len(text) > MAX_PAGE_CHARS:
            text = text[:MAX_PAGE_CHARS] + f"\n\n[… truncated at {MAX_PAGE_CHARS} chars]"
        return f"=== {path} — page {n}/{len(pages)} ===\n{text}"

    # Default: the structural summary + main-content boundary.
    s = paper_review.pdf_main_text_summary(target)
    if not s["ok"]:
        return f"Could not extract text from {path} (image-only PDF or pypdf missing)."

    total, main = s["total"], s["main_pages"]
    refs, appx = s["refs_page"], s["appendix_page"]
    lines = [f"=== {path} — {total} page(s) ==="]
    if main is not None and main != total:
        lines.append(f"Main content: ~{main} page(s) — ends on p.{main} "
                     f"(References + Appendix after it do NOT count toward the page limit).")
    else:
        lines.append(f"Main content: all {total} page(s) — no References/Appendix heading detected.")
    if refs:
        lines.append(f"References begin on p.{refs}.")
    if appx:
        lines.append(f"Appendix begins on p.{appx}.")
    lines.append("")
    for i, excerpt in s["per_page"]:
        tag = ""
        if i == refs:
            tag = "  ◀ References start"
        if i == appx:
            tag += "  ◀ Appendix start"
        lines.append(f" p.{i}: {excerpt}{tag}")
    return "\n".join(lines)
