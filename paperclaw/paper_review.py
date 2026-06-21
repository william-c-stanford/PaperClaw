"""Deterministic paper compliance review — the *verify* half of the
compile → verify → refine loop for venue (AAAI-style) submissions.

No OCR is needed: we compile the `.tex`, read the resulting PDF back with
``pypdf``, lint the source against the venue's hard rules (disallowed packages /
commands, required class/style), scan the compiler log for margin overflow and
undefined refs, and check the page limit. The agent then fixes the `.tex` and
re-reviews until it passes. This catches the *actual* AAAI formatting violations
far more reliably than reading rendered pixels.
"""

import re
from pathlib import Path

from paperclaw.iterative_pipeline import compile_tex

# AAAI camera-ready: 7 pages of technical content (refs / checklist excluded).
DEFAULT_PAGE_LIMIT = 7

# Packages an AAAI-style submission must NOT load (they fight the venue .sty);
# mirrors the disallowed list in the AAAI template / venue STYLE.md.
DISALLOWED_PACKAGES = {
    "authblk", "balance", "cjk", "float", "fontenc", "fullpage", "geometry",
    "hyperref", "multicol", "setspace", "titlesec", "ulem", "wrapfig",
}
# Commands the AAAI style forbids (page faking / spacing hacks).
DISALLOWED_COMMANDS = [
    r"\nocopyright", r"\addtolength", r"\balance", r"\baselinestretch",
    r"\clearpage", r"\newpage", r"\pagebreak", r"\pagestyle", r"\baselineskip=",
]

_PKG_RE = re.compile(r"\\usepackage(?:\[[^\]]*\])?\{([^}]*)\}")
_OVERFULL_RE = re.compile(r"Overfull \\[hv]box \(([\d.]+)pt too (?:wide|high)")
_COMMENT_RE = re.compile(r"(?<!\\)%.*")
_PAGELIMIT_RE = re.compile(r"up to (\d+)\s*pages", re.IGNORECASE)
_OVERFULL_MARGIN_PT = 5.0  # > this spills visibly into the margin

# Headings that begin the content EXCLUDED from the page limit (references +
# appendix). Matched as a standalone heading line, optionally section-numbered, so
# a stray "see references therein" in body text isn't mistaken for the section.
_REFS_HEAD_RE = re.compile(r"^\s*(?:\d+\s+)?(references|bibliography)\s*$",
                           re.IGNORECASE | re.MULTILINE)
_APPENDIX_HEAD_RE = re.compile(
    r"^\s*(?:appendix|appendices|supplementary(?:\s+material)?|technical\s+appendix)\b",
    re.IGNORECASE | re.MULTILINE)


def _strip_comments(src: str) -> str:
    return "\n".join(_COMMENT_RE.sub("", line) for line in src.splitlines())


def _pdf_pages_text(pdf_path: Path) -> list[str]:
    """Per-page extracted text (one string per page), or [] if unreadable."""
    try:
        from pypdf import PdfReader
        return [(p.extract_text() or "") for p in PdfReader(str(pdf_path)).pages]
    except Exception:
        return []


def _pdf_text(pdf_path: Path) -> str:
    return "\n".join(_pdf_pages_text(pdf_path))


def _extra_start_page(pages_text: list[str]) -> int | None:
    """1-based index of the first page that begins the References or Appendix —
    the content the page limit EXCLUDES — or None if neither is found."""
    refs = appx = None
    for i, t in enumerate(pages_text, start=1):
        if refs is None and _REFS_HEAD_RE.search(t):
            refs = i
        if appx is None and _APPENDIX_HEAD_RE.search(t):
            appx = i
    found = [p for p in (refs, appx) if p]
    return min(found) if found else None


def _main_text_pages(pages_text: list[str], total: int | None) -> tuple[int | None, int | None]:
    """(main_text_page_count, extra_start_page). The main text is everything before
    References/Appendix. If that section starts mid-page (right after the conclusion —
    likely, since `\\newpage` is disallowed), the main text shares that page, so it
    COUNTS; only when the page is essentially references/appendix from the top does it
    not. Falls back to *total* when no boundary is found."""
    extra = _extra_start_page(pages_text)
    if not extra:
        return total, None
    page = pages_text[extra - 1]
    heads = [m.start() for m in (_REFS_HEAD_RE.search(page), _APPENDIX_HEAD_RE.search(page)) if m]
    pos = min(heads) if heads else 0
    # Meaningful main text precedes the heading on this page → it's a shared page.
    shares = pos > max(40, int(0.10 * len(page)))
    return max(1, extra if shares else extra - 1), extra


def pdf_main_text_summary(pdf_path: Path) -> dict:
    """Where the MAIN content ends in a RENDERED PDF (the question `review_paper`'s
    page check answers, exposed standalone for the `read_pdf` tool).

    Returns ``{"ok", "total", "main_pages", "refs_page", "appendix_page",
    "per_page": [(page, excerpt), …]}``. ``main_pages`` is the main-text page count
    (References + Appendix excluded); ``refs_page`` / ``appendix_page`` are 1-based
    page numbers where each begins (None if not found). ``ok`` is False when the PDF
    can't be read (missing / image-only / no pypdf)."""
    pages = _pdf_pages_text(Path(pdf_path))
    if not pages:
        return {"ok": False, "total": None, "main_pages": None, "refs_page": None,
                "appendix_page": None, "per_page": []}
    refs = appx = None
    per_page: list[tuple[int, str]] = []
    for i, t in enumerate(pages, start=1):
        if refs is None and _REFS_HEAD_RE.search(t):
            refs = i
        if appx is None and _APPENDIX_HEAD_RE.search(t):
            appx = i
        per_page.append((i, " ".join(t.split())[:110]))
    main_pages, _ = _main_text_pages(pages, len(pages))
    return {"ok": True, "total": len(pages), "main_pages": main_pages,
            "refs_page": refs, "appendix_page": appx, "per_page": per_page}


def _compile_error_summary(log: str) -> str:
    lines = [ln for ln in log.splitlines()
             if ln.startswith("!") or "Error" in ln or "Emergency" in ln
             or re.match(r".*:\d+:", ln)]
    return " | ".join(lines[:4]) if lines else "(see the compiler log)"


def page_limit_from_style(style_md: str | None) -> int:
    """Pull a page limit out of a venue STYLE.md ('up to N pages'), else default."""
    if style_md:
        m = _PAGELIMIT_RE.search(style_md)
        if m:
            return int(m.group(1))
    return DEFAULT_PAGE_LIMIT


def review_paper(
    work_dir: Path,
    tex_name: str = "paper.tex",
    page_limit: int | None = None,
    venue_style: bool = True,
) -> dict:
    """Compile *tex_name* and check it for venue compliance.

    Returns ``{"ok", "pages", "issues": [(severity, message)], "log_tail"}`` where
    ``severity`` is ``"error"`` (must fix) or ``"warn"`` (should fix). ``ok`` is
    True only when there are no errors.
    """
    work_dir = Path(work_dir)
    tex_path = work_dir / tex_name
    if not tex_path.is_file():
        return {"ok": False, "pages": None,
                "issues": [("error", f"{tex_name} not found in the workspace.")], "log_tail": ""}

    src = tex_path.read_text(encoding="utf-8", errors="ignore")
    body = _strip_comments(src)
    issues: list[tuple[str, str]] = []

    # 1. Static source lint FIRST (independent of compiling) — so disallowed packages
    #    are reported as "remove it", not as a confusing "X.sty not found" if they're
    #    also uninstalled.
    if venue_style:
        for m in _PKG_RE.finditer(body):
            for pkg in (p.strip().lower() for p in m.group(1).split(",")):
                if pkg in DISALLOWED_PACKAGES:
                    issues.append(("error", f"disallowed package `{pkg}` — the venue style "
                                            f"forbids \\usepackage{{{pkg}}}; remove it."))
        for cmd in DISALLOWED_COMMANDS:
            if cmd.endswith("="):
                hit = cmd in body
            else:
                hit = re.search(re.escape(cmd) + r"(?![A-Za-z])", body) is not None
            if hit:
                issues.append(("error", f"disallowed command `{cmd}` — the venue style "
                                        f"forbids it (no page/spacing hacks)."))
        low_src = body.lower()
        if "aaai" in low_src and "[submission]" not in body and "[camera]" not in low_src:
            issues.append(("warn", "AAAI style not loaded with [submission] — use "
                                   "`\\usepackage[submission]{aaai2026}` for review."))

    # 2. Compile — everything below assumes a PDF.
    ok, pages, log = compile_tex(work_dir, tex_name)
    if not ok:
        issues.insert(0, ("error", "LaTeX did not compile — fix this first. "
                                   + _compile_error_summary(log)))
        return {"ok": False, "pages": pages, "issues": issues, "log_tail": log[-2000:]}

    # 3. Page limit — counts MAIN TEXT ONLY (references + appendix are excluded,
    #    as virtually every venue specifies). Find where the references/appendix
    #    begin in the rendered PDF and measure the main text up to that page.
    pages_text = _pdf_pages_text(work_dir / f"{tex_path.stem}.pdf")
    main_pages, extra_start = _main_text_pages(pages_text, pages) if pages_text else (pages, None)
    limit = page_limit if page_limit is not None else DEFAULT_PAGE_LIMIT
    if main_pages is not None and main_pages > limit:
        extra_note = (f" ({pages} pages total; references/appendix begin on p.{extra_start}, "
                      f"which don't count)" if extra_start else "")
        issues.append(("error", f"{main_pages} pages of main text exceeds the {limit}-page limit "
                                f"(main text only — references & appendix are excluded)"
                                f"{extra_note} — cut/condense {main_pages - limit} page(s)."))

    # 4. Compiler-log signals: margin overflow, undefined refs/cites, missing files.
    overs = [float(x) for x in _OVERFULL_RE.findall(log)]
    bad = [x for x in overs if x > _OVERFULL_MARGIN_PT]
    if bad:
        issues.append(("warn", f"{len(bad)} overfull box(es) up to {max(bad):.0f}pt — text/figures "
                               f"spill into the margin; rewrap, resize, or use \\resizebox."))
    if re.search(r"Citation [`'].*?' .*undefined", log):
        issues.append(("warn", "undefined citation(s) — add them to ref.bib with the `cite` tool "
                               "and \\cite the keys (no '?' marks in the PDF)."))
    if re.search(r"Reference [`'].*?' .*undefined", log) or "may have changed" in log:
        issues.append(("warn", "undefined \\ref/\\label or labels not yet resolved (recompile)."))
    if re.search(r"File [`'][^']+' not found", log):
        issues.append(("error", "an \\includegraphics / \\input file is missing — fix the path "
                                "or generate the figure."))

    # 5. Sanity-check structure from the page text already extracted above.
    text = "\n".join(pages_text).lower()
    if text:
        if "abstract" not in text:
            issues.append(("warn", "no 'Abstract' visible in the rendered PDF."))
        if not re.search(r"references|bibliography", text):
            issues.append(("warn", "no References section visible in the rendered PDF — "
                                   "end with \\bibliography{ref}."))
    else:
        issues.append(("warn", "could not read text from the PDF (scanned/empty?) — verify it renders."))

    ok_overall = not any(sev == "error" for sev, _ in issues)
    return {"ok": ok_overall, "pages": pages, "main_pages": main_pages,
            "extra_start": extra_start, "issues": issues, "log_tail": log[-1200:]}


def format_report(report: dict, tex_name: str = "paper.tex") -> str:
    """Render a review dict as a compact text report for the agent to act on."""
    pages = report.get("pages")
    main_pages = report.get("main_pages")
    head = f"Review of {tex_name}"
    if pages is not None:
        head += f" — {pages} pages"
        if main_pages is not None and main_pages != pages:
            head += f" (≈{main_pages} main text; references/appendix don't count)"
    errors = [m for sev, m in report["issues"] if sev == "error"]
    warns = [m for sev, m in report["issues"] if sev == "warn"]
    if report["ok"] and not warns:
        return f"{head}\n✓ COMPLIANT — no issues found. The paper meets the venue rules."
    lines = [head]
    lines.append("✓ COMPLIANT (warnings only)" if report["ok"]
                 else "✗ NOT COMPLIANT — fix the errors below, then re-review.")
    for m in errors:
        lines.append(f"  ✗ {m}")
    for m in warns:
        lines.append(f"  ! {m}")
    return "\n".join(lines)
