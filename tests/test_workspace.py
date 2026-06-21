"""Tests for the workspace file browser (list + raw serve) — what makes the
agent's code/figures/results viewable and inline figures render."""

import pytest
from fastapi.testclient import TestClient

from paperclaw.server.app import create_app


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    for var in ("PAPERCLAW_PROVIDER", "PAPERCLAW_BASE_URL", "PAPERCLAW_MODEL", "PAPERCLAW_API_KEY",
                "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


def _make_idea(client: TestClient) -> str:
    return client.post("/api/ideas", json={"title": "T"}).json()["id"]


def test_workspace_list_and_raw(tmp_path):
    client = TestClient(create_app(home=tmp_path))
    iid = _make_idea(client)

    base = tmp_path / "ideas" / iid / "hypotheses" / "H2"
    base.mkdir(parents=True)
    (base / "run.py").write_text("print('hi')\n", encoding="utf-8")
    (base / "fig1.png").write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
    (base / "__pycache__").mkdir()
    (base / "__pycache__" / "x.pyc").write_bytes(b"x")

    # whole-workspace listing, caches skipped
    r = client.get(f"/api/ideas/{iid}/files")
    assert r.status_code == 200
    paths = [e["path"] for e in r.json()["entries"]]
    assert "hypotheses/H2/run.py" in paths and "hypotheses/H2/fig1.png" in paths
    assert not any("__pycache__" in p for p in paths)

    # scoped listing
    r = client.get(f"/api/ideas/{iid}/files", params={"path": "hypotheses/H2"})
    assert r.status_code == 200 and r.json()["root"] == "hypotheses/H2"

    # raw image — correct content-type + bytes
    r = client.get(f"/api/ideas/{iid}/raw", params={"path": "hypotheses/H2/fig1.png"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/png")
    assert r.content.startswith(b"\x89PNG")

    # raw text
    r = client.get(f"/api/ideas/{iid}/raw", params={"path": "hypotheses/H2/run.py"})
    assert r.status_code == 200 and "print('hi')" in r.text


def test_workspace_path_escape_blocked(tmp_path):
    client = TestClient(create_app(home=tmp_path))
    iid = _make_idea(client)
    assert client.get(f"/api/ideas/{iid}/raw",
                      params={"path": "../../etc/passwd"}).status_code == 404
    assert client.get(f"/api/ideas/{iid}/files",
                      params={"path": "../.."}).status_code == 404


def test_workspace_missing(tmp_path):
    client = TestClient(create_app(home=tmp_path))
    assert client.get("/api/ideas/nope/files").status_code == 404
    assert client.get("/api/ideas/nope/raw", params={"path": "x"}).status_code == 404
