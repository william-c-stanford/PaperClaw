"""Download a domain's reference codebase from GitHub (no `git` dependency).

A domain can own one canonical, runnable reference implementation. We fetch it as a
tarball (GitHub's `api.github.com/repos/<owner>/<repo>/tarball/<ref>`, which redirects
to the default-branch tarball when `<ref>` is empty) and extract it into the domain's
`codebase/` dir, stripping the single top-level folder GitHub wraps the repo in.
Experiments for ideas pinned to that domain then read/import/copy from it.
"""

import io
import re
import shutil
import tarfile
from pathlib import Path

import httpx

_GITHUB_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/#?]+?)(?:\.git)?"
    r"(?:/tree/(?P<ref>[^/#?]+))?/?$", re.IGNORECASE)

_DOWNLOAD_TIMEOUT = 120
_MAX_TARBALL_BYTES = 200 * 1024 * 1024   # 200 MB compressed cap
_MAX_FILES = 20000
_MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024  # 1 GB extracted cap


class CodebaseError(Exception):
    """Raised when a codebase URL can't be parsed or downloaded."""


def resolve_tarball_url(url: str) -> tuple[str, str, str, str]:
    """Parse a GitHub repo URL → (owner, repo, ref, tarball_url).

    Accepts `https://github.com/owner/repo[.git]` and `.../tree/<branch-or-tag>`.
    `ref` is "" for the default branch. Raises CodebaseError on anything else
    (only GitHub is supported for now)."""
    m = _GITHUB_RE.match((url or "").strip())
    if not m:
        raise CodebaseError(
            "only GitHub repo URLs are supported, e.g. https://github.com/<owner>/<repo>")
    owner, repo, ref = m.group("owner"), m.group("repo"), m.group("ref") or ""
    tarball = f"https://api.github.com/repos/{owner}/{repo}/tarball/{ref}"
    return owner, repo, ref, tarball


def _safe_members(tar: tarfile.TarFile, dest: Path):
    """Yield (member, relative_path) for files under the single top-level dir,
    guarding against path traversal, the `.git/` tree, and zip-bombs."""
    total = 0
    files = 0
    for member in tar.getmembers():
        if not (member.isfile() or member.isdir()):
            continue  # skip symlinks/devices from an untrusted tarball
        parts = Path(member.name).parts
        if len(parts) <= 1:
            continue  # the wrapping top-level dir itself
        rel = Path(*parts[1:])
        if ".git" in rel.parts:
            continue
        target = (dest / rel).resolve()
        try:
            target.relative_to(dest.resolve())
        except ValueError:
            continue  # path traversal attempt — skip
        if member.isfile():
            files += 1
            total += member.size
            if files > _MAX_FILES or total > _MAX_UNCOMPRESSED_BYTES:
                raise CodebaseError("reference codebase is too large to download")
        yield member, rel


def download_codebase(url: str, dest_dir: Path) -> dict:
    """Download the GitHub repo at *url* into *dest_dir* (cleared first).

    Returns ``{"url", "ref", "fileCount"}``. Raises CodebaseError on failure."""
    owner, repo, ref, tarball = resolve_tarball_url(url)
    try:
        resp = httpx.get(tarball, timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True,
                         headers={"Accept": "application/vnd.github+json"})
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise CodebaseError(f"download failed for {owner}/{repo}: {exc}")
    data = resp.content
    if len(data) > _MAX_TARBALL_BYTES:
        raise CodebaseError("reference codebase tarball exceeds the size limit")

    dest_dir = Path(dest_dir)
    if dest_dir.exists():
        shutil.rmtree(dest_dir, ignore_errors=True)
    dest_dir.mkdir(parents=True, exist_ok=True)

    file_count = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member, rel in _safe_members(tar, dest_dir):
                target = dest_dir / rel
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    with open(target, "wb") as fh:
                        shutil.copyfileobj(extracted, fh)
                    file_count += 1
    except tarfile.TarError as exc:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise CodebaseError(f"could not extract the codebase tarball: {exc}")

    if file_count == 0:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise CodebaseError("the downloaded codebase was empty")
    return {"url": url, "ref": ref or "default", "fileCount": file_count}
