"""list_files tool — list the files in the research workspace.

The agent needs to know which files actually exist (their exact names) before
reading or editing them — otherwise it guesses names like 'PAPER.md' and fails.
The workspace is the idea or domain folder: the spec (IDEA.md / DOMAIN.md), the
generated paper (paper.md) if present, and hidden research artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SCHEMA: dict[str, Any] = {
    "name": "list_files",
    "description": (
        "List the files in the current research workspace (the idea or domain "
        "folder) with their sizes, so you know the EXACT file names that exist "
        "before reading or editing them. The workspace may contain the spec "
        "(IDEA.md or DOMAIN.md), the generated paper (paper.md), and research "
        "artifacts. Call this first when the user refers to a file you have not "
        "seen yet (e.g. 'the paper')."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
    },
}


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def execute(base_dir: Path, inputs: dict[str, Any]) -> str:
    """Return a newline list of workspace files (recursive, relative paths)."""
    base = base_dir.resolve()
    if not base.is_dir():
        return "Error: workspace folder does not exist."

    files = sorted(
        (p for p in base.rglob("*") if p.is_file()),
        key=lambda p: str(p.relative_to(base)).lower(),
    )
    if not files:
        return "Workspace is empty."

    lines = [f"Files in workspace ({len(files)}):"]
    for p in files:
        rel = p.relative_to(base)
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        lines.append(f"- {rel}  ({_fmt_size(size)})")
    return "\n".join(lines)
