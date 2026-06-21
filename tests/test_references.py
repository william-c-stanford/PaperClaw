"""Tests for ref.bib management: BibTeX parse/build, validation, routes, cite tool."""

from pathlib import Path

from fastapi.testclient import TestClient

from paperclaw import references as R
from paperclaw.server.app import create_app


def make_client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(home=tmp_path))


# ── cite keys & BibTeX ────────────────────────────────────────────────────────

def test_cite_key_scholar_style():
    assert R.cite_key(["Ashish Vaswani", "Noam Shazeer"], 2017, "Attention Is All You Need") == "vaswani2017attention"
    # stopwords (On/the/of) skipped → first significant word
    assert R.cite_key(["Doe, Jane"], None, "On the Theory of Minds") == "doendtheory"


def test_bibtex_roundtrip():
    bib = R.bibtex_from_paper({
        "title": "Deep Models", "authors": ["Jane Doe", "Bob Roe"],
        "year": 2020, "venue": "Nature", "doi": "10.1/x",
    })
    e = R.parse_bibtex(bib)[0]
    assert e["key"] == "doe2020deep"
    assert e["title"] == "Deep Models"
    assert e["authors"] == ["Jane Doe", "Bob Roe"]
    assert e["year"] == 2020 and e["doi"] == "10.1/x" and e["venue"] == "Nature"


def test_parse_nested_braces():
    cr = '@article{x2020, title = {A {Nested} Title}, author = {Smith, J and Doe, A}, year = {2020}, doi = {10.1/z}}'
    e = R.parse_bibtex(cr)[0]
    assert e["title"] == "A {Nested} Title"
    assert e["authors"] == ["Smith, J", "Doe, A"]
    assert e["year"] == 2020 and e["doi"] == "10.1/z"


# ── validation (network helpers monkeypatched) ────────────────────────────────

def test_validate_by_doi(monkeypatch):
    monkeypatch.setattr(R, "_crossref_title_for_doi", lambda doi: "Attention Is All You Need")
    assert R.validate_entry({"key": "k", "title": "attention is all you need", "doi": "10.x"})["status"] == "VERIFIED"

    monkeypatch.setattr(R, "_crossref_title_for_doi", lambda doi: "A Completely Unrelated Paper")
    assert R.validate_entry({"key": "k", "title": "attention is all you need", "doi": "10.x"})["status"] == "MISMATCH"

    monkeypatch.setattr(R, "_crossref_title_for_doi", lambda doi: None)
    assert R.validate_entry({"key": "k", "title": "x", "doi": "10.x"})["status"] == "NOT_FOUND"


def test_validate_by_title(monkeypatch):
    monkeypatch.setattr(R, "_crossref_search_title", lambda t: ("Attention Is All You Need", "10.x"))
    assert R.validate_entry({"key": "k", "title": "attention is all you need", "doi": None})["status"] == "VERIFIED"

    monkeypatch.setattr(R, "_crossref_search_title", lambda t: None)
    monkeypatch.setattr(R.literature, "search_papers_sync", lambda *a, **k: [])
    assert R.validate_entry({"key": "k", "title": "A Fabricated Paper", "doi": None})["status"] == "NOT_FOUND"


# ── routes ────────────────────────────────────────────────────────────────────

def test_reference_routes(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    idea_id = client.post("/api/ideas", json={"title": "T"}).json()["id"]

    assert client.get(f"/api/ideas/{idea_id}/references").json()["entries"] == []

    monkeypatch.setattr(
        "paperclaw.references.build_entry",
        lambda doi=None, query=None: "@article{doe2020thing, title={A Thing}, author={Jane Doe}, year={2020}}",
    )
    r = client.post(f"/api/ideas/{idea_id}/references/add", json={"query": "a thing"})
    assert r.status_code == 200
    entries = r.json()["entries"]
    assert entries and entries[0]["key"] == "doe2020thing"

    # dedup: adding the same key again does not duplicate
    client.post(f"/api/ideas/{idea_id}/references/add", json={"query": "a thing"})
    assert len(client.get(f"/api/ideas/{idea_id}/references").json()["entries"]) == 1

    monkeypatch.setattr(
        "paperclaw.references.validate_all",
        lambda es: [{"key": e["key"], "status": "VERIFIED", "detail": "ok"} for e in es],
    )
    v = client.post(f"/api/ideas/{idea_id}/references/validate").json()
    assert v and v[0]["status"] == "VERIFIED"


def test_add_reference_not_found(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    idea_id = client.post("/api/ideas", json={"title": "T"}).json()["id"]
    monkeypatch.setattr("paperclaw.references.build_entry", lambda doi=None, query=None: None)
    r = client.post(f"/api/ideas/{idea_id}/references/add", json={"query": "nonexistent"})
    assert r.status_code == 422


# ── cite tool ─────────────────────────────────────────────────────────────────

def test_cite_tool(tmp_path, monkeypatch):
    from paperclaw.tools import cite
    monkeypatch.setattr(
        "paperclaw.references.build_entry",
        lambda doi=None, query=None: "@article{k2020thing, title={Thing}, author={A B}, year={2020}}",
    )
    out = cite.execute(tmp_path, {"query": "thing"})
    assert "k2020thing" in out and (tmp_path / "ref.bib").is_file()
    assert "Already in ref.bib" in cite.execute(tmp_path, {"query": "thing"})  # dedup
