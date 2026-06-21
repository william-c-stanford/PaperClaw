"""write_file tool — create a new file or overwrite a whole file.

Companion to apply_patch: apply_patch makes targeted edits to an EXISTING file,
while write_file creates a file that does not exist yet (e.g. the agent wants to
write paper.md from scratch) or replaces a whole file's contents. This is what
lets the agent create files instead of failing when apply_patch reports the
target is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SCHEMA: dict[str, Any] = {
    "name": "write_file",
    "description": (
        "Create a new file, or overwrite an existing file, in the research "
        "workspace with the full content you provide. Use this to CREATE a file "
        "(apply_patch only edits files that already exist) or to replace a whole "
        "file such as paper.md. For a small change to an existing file, prefer "
        "apply_patch instead of rewriting it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Relative path to the file from the idea or domain folder. "
                    "Examples: 'paper.md', 'notes.md', 'IDEA.md'."
                ),
            },
            "content": {
                "type": "string",
                "description": "The complete new contents of the file.",
            },
            "reason": {
                "type": "string",
                "description": "One-sentence explanation of what you are writing and why.",
            },
        },
        "required": ["path", "content"],
    },
}


def execute(base_dir: Path, inputs: dict[str, Any]) -> str:
    """Write *content* to *path* (relative to *base_dir*), creating parents."""
    path = inputs.get("path", "")
    content = inputs.get("content")
    reason = inputs.get("reason", "")

    if not path:
        return "Error: 'path' is required."
    if content is None:
        return "Error: 'content' is required."

    target = (base_dir / path).resolve()
    try:
        target.relative_to(base_dir.resolve())
    except ValueError:
        return f"Error: path escapes workspace: {path!r}"

    existed = target.exists()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"Error writing file: {exc}"

    verb = "Overwrote" if existed else "Created"
    n = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    prefix = f"[{reason}] " if reason else ""
    return f"{prefix}{verb} {path} ({n} lines)"
