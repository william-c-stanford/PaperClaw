"""apply_patch tool — apply a unified diff to a file in the research workspace.

Implements a self-contained unified-diff applier (no external dep) and the
Anthropic tool schema so it can be passed to messages.create(tools=[...]).

Public API
----------
apply_diff(original, diff)   → patched string  (pure transform)
apply_patch(base_dir, path, diff) → result string  (reads + writes file)
execute(base_dir, inputs)    → result string  (tool-call entry point)
SCHEMA                       → Anthropic tool_use JSON definition
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# ── Anthropic tool schema ─────────────────────────────────────────────────────

SCHEMA: dict[str, Any] = {
    "name": "apply_patch",
    "description": (
        "Apply a targeted patch to a file using a unified diff. "
        "Use this to make specific edits instead of rewriting the whole file. "
        "Works on DOMAIN.md, IDEA.md, and any text file in the research workspace."
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
            "diff": {
                "type": "string",
                "description": (
                    "Unified diff with @@ hunk headers. IMPORTANT: the patch is "
                    "located by MATCHING the context (' ') and removed ('-') lines "
                    "against the current file — the @@ line numbers may be "
                    "approximate. So include a few EXACT context lines around the "
                    "change (copied verbatim from read_file), enough to be unique. "
                    "Example:\n"
                    "@@ -3,4 +3,4 @@\n"
                    " ## Crucial Papers\n"
                    "-Old paper title (2020)\n"
                    "+New paper title (2024)\n"
                    " \n"
                    " ## Datasets"
                ),
            },
            "reason": {
                "type": "string",
                "description": "One-sentence explanation of what is changing and why.",
            },
        },
        "required": ["path", "diff"],
    },
}

# ── Diff parsing ──────────────────────────────────────────────────────────────

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _parse_hunks(diff: str) -> list[tuple[int, list[str]]]:
    """Return [(orig_start_1indexed, hunk_lines), …] for each @@ block."""
    hunks: list[tuple[int, list[str]]] = []
    cur_start = 0
    cur_lines: list[str] = []
    in_hunk = False

    for raw in diff.splitlines(keepends=True):
        m = _HUNK_RE.match(raw)
        if m:
            if in_hunk:
                hunks.append((cur_start, cur_lines))
            cur_start = int(m.group(1))
            cur_lines = []
            in_hunk = True
        elif in_hunk:
            if raw.startswith(("+", "-", " ", "\\")):
                cur_lines.append(raw)
            # skip trailing @@ comments or blank lines between hunks
        # skip --- / +++ header lines before first @@

    if in_hunk and cur_lines:
        hunks.append((cur_start, cur_lines))
    return hunks


def _hunk_blocks(hunk: list[str]) -> tuple[list[str], list[str]]:
    """Split a hunk into (old_block, new_block).

    old_block = the lines the hunk expects to FIND in the file (context + removed);
    new_block = the lines it should become (context + added). Both newline-terminated.
    """
    old: list[str] = []
    new: list[str] = []
    for raw in hunk:
        if not raw or raw.startswith("\\"):  # skip "\ No newline at end of file"
            continue
        op, content = raw[0], raw[1:]
        if not content.endswith("\n"):
            content += "\n"
        if op == " ":
            old.append(content)
            new.append(content)
        elif op == "-":
            old.append(content)
        elif op == "+":
            new.append(content)
    return old, new


def _find_block(lines: list[str], old: list[str], hint: int) -> int | None:
    """Index where *old* matches *lines* contiguously, nearest *hint*; None if not found.

    LLM-written diffs frequently carry WRONG @@ line numbers, so we locate the edit
    by its content (context + removed lines) rather than trusting the offset — the
    hint is only a tiebreaker when the block matches in more than one place. Tries
    exact match, then a trailing-whitespace-tolerant match.
    """
    n = len(old)
    if n == 0 or n > len(lines):
        return None
    for eq in (lambda a, b: a == b, lambda a, b: a.rstrip() == b.rstrip()):
        cands = [i for i in range(len(lines) - n + 1)
                 if all(eq(lines[i + j], old[j]) for j in range(n))]
        if cands:
            return min(cands, key=lambda i: abs(i - hint))
    return None


# ── Public transform ──────────────────────────────────────────────────────────

def apply_diff(original: str, diff: str) -> str:
    """Apply a unified diff to *original* and return the patched string.

    Edits are located by matching each hunk's context + removed lines against the
    file (NOT by the @@ line numbers, which LLMs get wrong). Raises ``ValueError``
    with an actionable message when a hunk's context can't be found — so a bad
    diff fails loudly instead of editing the wrong line.
    """
    hunks = _parse_hunks(diff)
    if not hunks:
        raise ValueError(
            "No hunk headers found — make sure the diff includes @@ -L,N +L,N @@ lines."
        )

    lines: list[str] = original.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"  # normalise trailing newline for consistent matching

    offset = 0  # tracks net line shift, used only to bias the match hint
    for idx, (orig_start, hunk_lines) in enumerate(hunks, 1):
        old, new = _hunk_blocks(hunk_lines)
        hint = max(0, orig_start - 1 + offset)
        if not old:
            # Pure insertion (no context/removed lines): fall back to the position
            # hint — there is nothing to match against.
            pos = min(max(hint, 0), len(lines))
            lines = lines[:pos] + new + lines[pos:]
            offset += len(new)
            continue
        start = _find_block(lines, old, hint)
        if start is None:
            raise ValueError(
                f"Could not locate hunk {idx} in the file — its context/removed "
                "lines don't match the current content (the diff's line numbers are "
                "ignored, only the content matters). Re-read the file with read_file "
                "and regenerate the diff with exact context lines."
            )
        lines = lines[:start] + new + lines[start + len(old):]
        offset += len(new) - len(old)

    result = "".join(lines)
    # Strip the normalisation newline we may have added
    if not original.endswith("\n") and result.endswith("\n"):
        result = result[:-1]
    return result


# ── File-level helper ─────────────────────────────────────────────────────────

def apply_patch(base_dir: Path, path: str, diff: str) -> str:
    """Read *path* (relative to *base_dir*), apply *diff*, write back.

    Returns a human-readable result string for the tool_result message.
    Raises ``ValueError`` / ``FileNotFoundError`` on failure.
    """
    target = (base_dir / path).resolve()
    # Safety: never escape the workspace
    try:
        target.relative_to(base_dir.resolve())
    except ValueError:
        raise ValueError(f"Path escapes workspace: {path!r}")

    if not target.exists():
        raise FileNotFoundError(f"File not found: {path}")

    original = target.read_text(encoding="utf-8")
    patched = apply_diff(original, diff)
    target.write_text(patched, encoding="utf-8")

    orig_n = original.count("\n")
    new_n = patched.count("\n")
    delta = new_n - orig_n
    sign = "+" if delta >= 0 else ""
    return f"Patched {path} ({orig_n} → {new_n} lines, {sign}{delta})"


# ── Tool executor (called by llm.py tool-loop) ────────────────────────────────

def execute(base_dir: Path, inputs: dict[str, Any]) -> str:
    """Entry point for the Anthropic tool-call loop.

    ``inputs`` is the parsed ``input`` field from a ``tool_use`` block.
    Returns the string that goes into the ``tool_result`` content.
    """
    path = inputs.get("path", "")
    diff = inputs.get("diff", "")
    reason = inputs.get("reason", "")

    if not path:
        return "Error: 'path' is required."
    if not diff:
        return "Error: 'diff' is required."

    try:
        result = apply_patch(base_dir, path, diff)
        prefix = f"[{reason}] " if reason else ""
        return prefix + result
    except (ValueError, FileNotFoundError, OSError) as exc:
        return f"Error applying patch: {exc}"
