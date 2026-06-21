"""Per-idea reference management: BibTeX generation, parsing, and validation.

Each idea owns a ``ref.bib`` (the citation source of truth). Entries are built
deterministically from OpenAlex (``literature.py``) and Crossref so the cite keys
are stable and Google-Scholar-compatible (``lastnameYEARword``) WITHOUT scraping
Google Scholar (which has no API and blocks crawlers).

Validation checks every entry against Crossref/OpenAlex to catch fabricated or
wrong citations — each entry is classified:
  VERIFIED  — found and the title matches
  MISMATCH  — found by DOI but the title differs (wrong/garbled citation)
  NOT_FOUND — no match anywhere (likely fabricated)
  UNKNOWN   — network/lookup failure (do NOT treat as fabricated)
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import httpx

from paperclaw import literature

_TIMEOUT = 12.0
_HEADERS = {"User-Agent": "PaperClaw (mailto:paperclaw@example.com)"}
_CROSSREF = "https://api.crossref.org/works"
_TITLE_MATCH = 0.85  # SequenceMatcher ratio above which two titles are "the same"

_STOPWORDS = {"a", "an", "the", "on", "of", "for", "and", "to", "in", "with", "via", "is"}


# ── Cite keys & BibTeX building ───────────────────────────────────────────────

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _last_name(author: str) -> str:
    author = author.strip()
    if "," in author:  # "Last, First"
        return author.split(",")[0]
    return author.split()[-1] if author else ""


def cite_key(authors: list[str], year: int | None, title: str) -> str:
    """Scholar-style key: firstauthorlastname + year + first significant title word."""
    last = _slug(_last_name(authors[0])) if authors else "anon"
    yr = str(year) if year else "nd"
    first_word = ""
    for w in re.findall(r"[A-Za-z0-9]+", title or ""):
        if w.lower() not in _STOPWORDS:
            first_word = w.lower()
            break
    return f"{last}{yr}{first_word}" or "ref"


def _bibtex_authors(authors: list[str]) -> str:
    return " and ".join(a.strip() for a in authors if a.strip()) or "Unknown"


def bibtex_from_paper(p: dict, key: str | None = None) -> str:
    """Build a BibTeX entry from a paper dict (OpenAlex shape)."""
    authors = p.get("authors") or []
    year = p.get("year")
    title = (p.get("title") or "").strip()
    venue = (p.get("venue") or "").strip()
    doi = (p.get("doi") or "").replace("https://doi.org/", "").strip()
    key = key or cite_key(authors, year, title)
    etype = "article" if venue else "misc"
    lines = [f"@{etype}{{{key},"]
    lines.append(f"  title = {{{title}}},")
    lines.append(f"  author = {{{_bibtex_authors(authors)}}},")
    if year:
        lines.append(f"  year = {{{year}}},")
    if venue:
        lines.append(f"  journal = {{{venue}}},")
    if doi:
        lines.append(f"  doi = {{{doi}}},")
    lines.append("}")
    return "\n".join(lines)


# ── BibTeX parsing ────────────────────────────────────────────────────────────

def _parse_fields(body: str) -> dict[str, str]:
    """Parse `name = {value}` / `name = "value"` / `name = bareword` pairs."""
    fields: dict[str, str] = {}
    i, n = 0, len(body)
    while i < n:
        m = re.compile(r"([A-Za-z_-]+)\s*=\s*").match(body, i)
        if not m:
            i += 1
            continue
        name = m.group(1).lower()
        i = m.end()
        if i >= n:
            break
        if body[i] == "{":
            depth, j = 0, i
            while j < n:
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            value = body[i + 1:j]
            i = j + 1
        elif body[i] == '"':
            j = body.find('"', i + 1)
            j = j if j != -1 else n
            value = body[i + 1:j]
            i = j + 1
        else:
            j = i
            while j < n and body[j] not in ",\n":
                j += 1
            value = body[i:j].strip()
            i = j
        fields[name] = re.sub(r"\s+", " ", value).strip()
        comma = body.find(",", i)
        i = comma + 1 if comma != -1 else n
    return fields


def _to_reference(etype: str, key: str, fields: dict[str, str]) -> dict:
    authors_raw = fields.get("author", "")
    authors = [a.strip() for a in re.split(r"\s+and\s+", authors_raw) if a.strip()]
    year = None
    if fields.get("year"):
        ym = re.search(r"\d{4}", fields["year"])
        year = int(ym.group()) if ym else None
    return {
        "key": key,
        "type": etype,
        "title": fields.get("title", "").strip(),
        "authors": authors,
        "year": year,
        "doi": fields.get("doi", "").replace("https://doi.org/", "").strip() or None,
        "venue": (fields.get("journal") or fields.get("booktitle") or "").strip() or None,
    }


def parse_bibtex(text: str) -> list[dict]:
    """Tolerant BibTeX parser → list of reference dicts (brace-balanced)."""
    entries: list[dict] = []
    i, n = 0, len(text or "")
    while True:
        at = text.find("@", i)
        if at == -1:
            break
        brace = text.find("{", at)
        if brace == -1:
            break
        etype = text[at + 1:brace].strip().lower()
        depth, j = 0, brace
        while j < n:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = text[brace + 1:j]
        i = j + 1
        comma = body.find(",")
        if comma == -1 or etype in ("comment", "preamble", "string"):
            continue
        key = body[:comma].strip()
        entries.append(_to_reference(etype, key, _parse_fields(body[comma + 1:])))
    return entries


def keys_in(text: str) -> set[str]:
    return {e["key"] for e in parse_bibtex(text)}


# ── Crossref lookups (sync; run in a worker thread by callers) ─────────────────

def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).strip()


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_title(a), _norm_title(b)).ratio()


def fetch_crossref_bibtex(doi: str) -> str | None:
    """DOI → BibTeX via content negotiation (works across registrars)."""
    doi = (doi or "").replace("https://doi.org/", "").strip()
    if not doi:
        return None
    try:
        with httpx.Client(timeout=_TIMEOUT, headers={**_HEADERS, "Accept": "application/x-bibtex"}) as c:
            r = c.get(f"https://doi.org/{doi}", follow_redirects=True)
        if r.status_code == 200 and "@" in r.text:
            return r.text.strip()
    except Exception:
        pass
    return None


def _crossref_title_for_doi(doi: str) -> str | None:
    try:
        with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS) as c:
            r = c.get(f"{_CROSSREF}/{doi}")
        if r.status_code == 200:
            titles = (r.json().get("message") or {}).get("title") or []
            return titles[0] if titles else ""
    except Exception:
        return None
    return None  # 404 → not found


def _crossref_search_title(title: str) -> tuple[str, str] | None:
    """Return (matched_title, doi) for the best Crossref hit, or None."""
    try:
        with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS) as c:
            r = c.get(_CROSSREF, params={"query.bibliographic": title, "rows": 3,
                                         "select": "DOI,title"})
        if r.status_code != 200:
            return None
        for item in (r.json().get("message") or {}).get("items", []):
            t = (item.get("title") or [""])[0]
            if t and _similar(title, t) >= _TITLE_MATCH:
                return t, item.get("DOI", "")
    except Exception:
        return None
    return None


def validate_entry(entry: dict) -> dict:
    """Classify one reference entry. Returns {key, status, detail}."""
    key, title, doi = entry["key"], entry.get("title", ""), entry.get("doi")
    if not title:
        return {"key": key, "status": "NOT_FOUND", "detail": "entry has no title"}

    if doi:
        found = _crossref_title_for_doi(doi)
        if found is None:
            return {"key": key, "status": "NOT_FOUND", "detail": f"DOI {doi} not found on Crossref"}
        if found == "":  # network/empty — fall through to title search
            pass
        elif _similar(title, found) >= _TITLE_MATCH:
            return {"key": key, "status": "VERIFIED", "detail": f"DOI matches: {found}"}
        else:
            return {"key": key, "status": "MISMATCH",
                    "detail": f"DOI {doi} is titled “{found}”, not “{title}”"}

    hit = _crossref_search_title(title)
    if hit:
        matched, found_doi = hit
        return {"key": key, "status": "VERIFIED",
                "detail": f"matched “{matched}”" + (f" (doi {found_doi})" if found_doi else "")}
    # last resort: OpenAlex title search
    try:
        papers = literature.search_papers_sync(title, limit=3)
        for p in papers:
            if _similar(title, p.get("title", "")) >= _TITLE_MATCH:
                return {"key": key, "status": "VERIFIED", "detail": f"matched on OpenAlex: {p['title']}"}
    except Exception:
        return {"key": key, "status": "UNKNOWN", "detail": "lookup failed (network)"}
    return {"key": key, "status": "NOT_FOUND", "detail": "no Crossref/OpenAlex match — likely fabricated"}


def validate_all(entries: list[dict]) -> list[dict]:
    return [validate_entry(e) for e in entries]


def build_entry(doi: str | None = None, query: str | None = None) -> str | None:
    """Build one BibTeX entry: prefer Crossref-by-DOI, else OpenAlex-by-query."""
    if doi:
        bib = fetch_crossref_bibtex(doi)
        if bib:
            return bib
        papers = literature.search_papers_sync(doi, limit=1)
        if papers:
            return bibtex_from_paper(papers[0])
        return None
    if query:
        papers = literature.search_papers_sync(query, limit=1)
        if papers:
            p = papers[0]
            if p.get("doi"):
                bib = fetch_crossref_bibtex(p["doi"])
                if bib:
                    return bib
            return bibtex_from_paper(p)
    return None
