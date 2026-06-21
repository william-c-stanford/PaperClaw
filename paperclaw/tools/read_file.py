"""read_file tool — read a file from the research workspace.

Companion to apply_patch: the agent reads the current state before patching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SCHEMA: dict[str, Any] = {
    "name": "read_file",
    "description": (
        "Read the current contents of a file in the research workspace. "
        "Use this before apply_patch to see what you are editing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Relative path to the file from the idea or domain folder. "
                    "Examples: 'DOMAIN.md', 'IDEA.md', '.research_plan.md'"
                ),
            },
        },
        "required": ["path"],
    },
}

MAX_CHARS = 12_000  # truncation guard for very large files


def execute(base_dir: Path, inputs: dict[str, Any]) -> str:
    """Return file contents (truncated if large) or an error string."""
    path = inputs.get("path", "")
    if not path:
        return "Error: 'path' is required."

    target = (base_dir / path).resolve()
    try:
        target.relative_to(base_dir.resolve())
    except ValueError:
        return f"Error: path escapes workspace: {path!r}"

    if not target.exists():
        return f"Error: file not found: {path}"

    try:
        content = target.read_text(encoding="utf-8")
    except OSError as exc:
        return f"Error reading file: {exc}"

    if len(content) > MAX_CHARS:
        content = content[:MAX_CHARS] + f"\n\n[… truncated at {MAX_CHARS} chars]"

    return f"=== {path} ===\n{content}"
