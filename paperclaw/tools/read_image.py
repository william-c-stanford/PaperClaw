"""read_image tool — let the (vision-capable) LLM actually SEE an image file.

`read_file` returns text and chokes on binary images; this returns the image as a
content block the vision model can look at — so the agent can inspect a result
figure (axis labels, legend, layout, any text baked into the PNG) instead of
guessing or rebuilding it blind. A short text summary (dimensions / size) is
always included, so a non-vision model still gets *something* useful.

Unlike most tools this may return a LIST of Anthropic content blocks (text +
image) rather than a plain string — the tool loops pass that straight through to
Anthropic; the OpenAI-compatible path flattens it to the text summary (tool-role
images aren't portable across those endpoints).
"""

from __future__ import annotations

import base64
import struct
from pathlib import Path
from typing import Any

SCHEMA: dict[str, Any] = {
    "name": "read_image",
    "description": (
        "View an image file (PNG/JPG/GIF/WebP) from the research workspace — e.g. a "
        "result figure or a venue template page. The image is shown to you directly so "
        "you can read its axis labels, legend, layout and any text baked into it. Use "
        "this to check a generated figure before referencing it in the paper "
        "(read_file cannot display binary images)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Relative path to the image from the idea or domain folder. "
                    "Examples: 'experiments/1/loss.png', 'fig_method.png'."
                ),
            },
        },
        "required": ["path"],
    },
}

# Anthropic caps a single image near 5 MB of base64; guard the raw bytes well under.
MAX_BYTES = 5 * 1024 * 1024
_MEDIA = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _png_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from a PNG's IHDR chunk, or None if not a parseable PNG."""
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        w, h = struct.unpack(">II", data[16:24])
        return w, h
    return None


def execute(base_dir: Path, inputs: dict[str, Any]) -> str | list[dict[str, Any]]:
    """Return [text, image] content blocks for the model, or an error string."""
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

    media = _MEDIA.get(target.suffix.lower())
    if media is None:
        return f"Error: not a supported image (PNG/JPG/GIF/WebP): {path}"

    try:
        raw = target.read_bytes()
    except OSError as exc:
        return f"Error reading image: {exc}"

    if len(raw) > MAX_BYTES:
        return (
            f"Error: image too large to view ({len(raw) // 1024} KB > "
            f"{MAX_BYTES // 1024} KB): {path}"
        )

    size = _png_size(raw)
    dims = f"{size[0]}×{size[1]}px, " if size else ""
    summary = f"{path} ({dims}{len(raw) // 1024} KB)"
    b64 = base64.b64encode(raw).decode("ascii")
    return [
        {"type": "text", "text": f"Image {summary}:"},
        {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
    ]
