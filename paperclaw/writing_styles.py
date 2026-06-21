"""Writing-style library — prose-style guides (tone, structure, phrasing) chosen at
paper-writing time. SEPARATE from venue formatting (`venue/STYLE.md` / `.sty`),
which governs layout, not how the prose reads.

Two scopes, resolved domain-first:
  - global   `<home>/writing_styles/<name>.md`        (domain-agnostic, shared)
  - domain   `domains/<id>/writing_styles/<name>.md`  (overrides a global of the same name)

The global library is seeded with a few built-in guides on first use.
"""

import re
from pathlib import Path

from paperclaw.prompts.writing_styles import BUILTIN_STYLES  # built-in guides

_NAME_RE = re.compile(r"[^a-z0-9._-]+")
_HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)

DIRNAME = "writing_styles"

__all__ = [
    "BUILTIN_STYLES", "DIRNAME", "ensure_seeded",
    "list_styles", "get_style", "save_style",
]


def _safe_name(name: str) -> str | None:
    n = _NAME_RE.sub("-", (name or "").strip().lower()).strip("-")
    return n or None


def _title(md: str, fallback: str) -> str:
    m = _HEADING_RE.search(md or "")
    return m.group(1).strip() if m else fallback


def ensure_seeded(home: Path) -> None:
    """Write the built-in guides into `<home>/writing_styles/` if missing."""
    d = Path(home) / DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    for name, md in BUILTIN_STYLES.items():
        f = d / f"{name}.md"
        if not f.exists():
            f.write_text(md, encoding="utf-8")


def _scan(d: Path, scope: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if d.is_dir():
        for f in sorted(d.glob("*.md")):
            out[f.stem] = {"name": f.stem, "scope": scope,
                           "title": _title(f.read_text(encoding="utf-8", errors="ignore"), f.stem)}
    return out


def list_styles(home: Path, domain_dir: Path | None = None) -> list[dict]:
    """All available styles — global + domain (domain overrides a same-named global)."""
    ensure_seeded(home)
    styles = _scan(Path(home) / DIRNAME, "global")
    if domain_dir is not None:
        styles.update(_scan(Path(domain_dir) / DIRNAME, "domain"))  # domain wins
    return sorted(styles.values(), key=lambda s: (s["scope"] != "domain", s["name"]))


def get_style(home: Path, domain_dir: Path | None, name: str) -> str | None:
    """Resolve a style guide's markdown by name — domain dir first, then global."""
    safe = _safe_name(name)
    if not safe:
        return None
    candidates = []
    if domain_dir is not None:
        candidates.append(Path(domain_dir) / DIRNAME / f"{safe}.md")
    ensure_seeded(home)
    candidates.append(Path(home) / DIRNAME / f"{safe}.md")
    for c in candidates:
        if c.is_file():
            return c.read_text(encoding="utf-8", errors="ignore")
    return None


def save_style(home: Path, domain_dir: Path | None, name: str, content: str) -> str | None:
    """Create/overwrite a style guide (domain-scoped if domain_dir given). Returns
    the saved name, or None if the name is invalid."""
    safe = _safe_name(name)
    if not safe:
        return None
    base = Path(domain_dir) if domain_dir is not None else Path(home)
    d = base / DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{safe}.md").write_text(content, encoding="utf-8")
    return safe
