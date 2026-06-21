"""Tests for the hypothesis map (routes + edit tools) and references generation."""

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from paperclaw import literature, llm
from paperclaw.server.app import create_app
from paperclaw.tools import hypothesis as H

MAP_JSON = ('```json\n{"nodes":[{"statement":"R1","rationale":"why",'
            '"children":[{"statement":"sub","test":"acc"}]}]}\n```')


def make_client(tmp_path):
    return TestClient(create_app(home=tmp_path))


def test_hypothesis_map_routes(tmp_path, monkeypatch):
    async def fake_chat(settings, system, messages, max_tokens=2048):
        return SimpleNamespace(text=MAP_JSON, model="x")
    monkeypatch.setattr(llm, "chat", fake_chat)

    c = make_client(tmp_path)
    i = c.post("/api/ideas", json={"title": "T"}).json()["id"]
    assert c.get(f"/api/ideas/{i}/hypothesis-map").json()["nodes"] == []

    r = c.post(f"/api/ideas/{i}/hypothesis-map/generate")
    assert r.status_code == 200
    nodes = r.json()["nodes"]
    assert nodes[0]["statement"] == "R1"
    assert nodes[0]["children"] == []  # roots-only: any LLM-proposed children are stripped
    assert nodes[0]["id"] and nodes[0]["status"] == "untested"


def test_expand_requires_verdicted_parent(tmp_path, monkeypatch):
    """A child sub-hypothesis is never grown under an unverdicted/blocked parent."""
    import asyncio
    from types import SimpleNamespace

    from paperclaw import service
    from paperclaw.config import LLMSettings
    from paperclaw.server.store import Store

    async def fake_chat(settings, system, messages, max_tokens=600):
        return SimpleNamespace(text='```json\n{"nodes":[{"statement":"child","test":"t"}]}\n```', model="x")
    monkeypatch.setattr(llm, "chat", fake_chat)

    store = Store(tmp_path)
    idea = store.add_idea("T")
    store.put_hypothesis_map(idea.id, {"nodes": [{"id": "H1", "statement": "R", "status": "untested", "children": []}]})

    # untested parent → refuse to expand
    assert asyncio.run(service.expand_hypothesis(store, LLMSettings(), idea.id, "H1", "rep")) == 0
    assert store.get_hypothesis_map(idea.id)["nodes"][0]["children"] == []

    # INCONCLUSIVE parent (e.g. experiment produced no usable results) → no expansion,
    # so a failed run never spawns degenerate "did the pipeline run?" meta-hypotheses
    store.put_hypothesis_map(idea.id, {"nodes": [{"id": "H1", "statement": "R", "status": "inconclusive", "children": []}]})
    assert asyncio.run(service.expand_hypothesis(store, LLMSettings(), idea.id, "H1", "rep")) == 0
    assert store.get_hypothesis_map(idea.id)["nodes"][0]["children"] == []

    # verdicted (supported) parent → expansion proceeds
    store.put_hypothesis_map(idea.id, {"nodes": [{"id": "H1", "statement": "R", "status": "supported", "children": []}]})
    assert asyncio.run(service.expand_hypothesis(store, LLMSettings(), idea.id, "H1", "rep")) == 1
    kids = store.get_hypothesis_map(idea.id)["nodes"][0]["children"]
    assert kids and kids[0]["id"] == "H1.1"


def test_expand_grows_sideways_at_max_depth(tmp_path, monkeypatch):
    """At the depth cap, expansion adds a SIBLING (under the parent), not a deeper child;
    without a cap the same expansion goes one level deeper."""
    import asyncio
    from types import SimpleNamespace

    from paperclaw import service
    from paperclaw.config import LLMSettings
    from paperclaw.server.store import Store

    async def fake_chat(settings, system, messages, max_tokens=600):
        return SimpleNamespace(text='```json\n{"nodes":[{"statement":"sib","test":"t"}]}\n```', model="x")
    monkeypatch.setattr(llm, "chat", fake_chat)

    store = Store(tmp_path)
    idea = store.add_idea("T")
    depth2_tree = lambda: {"nodes": [  # noqa: E731 — H1 → H1.1 (depth 2, supported)
        {"id": "H1", "statement": "R", "status": "supported", "expanded": True, "children": [
            {"id": "H1.1", "statement": "C", "status": "supported", "children": []}]}]}

    # cap depth at 2: expanding H1.1 (depth 2) grows a SIBLING H1.2, not a child H1.1.1
    store.put_hypothesis_map(idea.id, depth2_tree())
    assert asyncio.run(service.expand_hypothesis(store, LLMSettings(), idea.id, "H1.1", "rep", max_depth=2)) == 1
    h1 = store.get_hypothesis_map(idea.id)["nodes"][0]
    assert h1["children"][0]["children"] == []                 # H1.1 gained no deeper child
    assert any(c["id"] == "H1.2" for c in h1["children"])       # sibling added at the same depth

    # no cap → the same expansion goes DEEPER (a child H1.1.1)
    store.put_hypothesis_map(idea.id, depth2_tree())
    asyncio.run(service.expand_hypothesis(store, LLMSettings(), idea.id, "H1.1", "rep", max_depth=None))
    h11 = store.get_hypothesis_map(idea.id)["nodes"][0]["children"][0]
    assert h11["children"] and h11["children"][0]["id"] == "H1.1.1"


def test_delete_hypothesis_node_route(tmp_path):
    c = make_client(tmp_path)
    i = c.post("/api/ideas", json={"title": "T"}).json()["id"]
    c.app.state.store.put_hypothesis_map(i, {"nodes": [
        {"id": "H1", "statement": "R", "status": "supported", "children": [
            {"id": "H1.1", "statement": "c", "status": "untested", "children": []}]}]})

    # delete the child → root remains, child gone
    r = c.delete(f"/api/ideas/{i}/hypotheses/H1.1")
    assert r.status_code == 200
    nodes = r.json()["nodes"]
    assert nodes[0]["id"] == "H1" and nodes[0]["children"] == []

    # delete the root → empty map; missing id → 404
    assert c.delete(f"/api/ideas/{i}/hypotheses/H1").json()["nodes"] == []
    assert c.delete(f"/api/ideas/{i}/hypotheses/H9").status_code == 404


def test_map_generation_creates_root_plans_and_plan_endpoint(tmp_path, monkeypatch):
    async def fake_chat(settings, system, messages, max_tokens=2048):
        if "HYPOTHESIS MAP" in system:
            return SimpleNamespace(text=MAP_JSON, model="x")
        if "TESTING PLAN" in system:
            return SimpleNamespace(text="```plan\n## Datasets\nMNIST\n## Feasibility\nFEASIBLE\n```", model="x")
        return SimpleNamespace(text="q", model="x")
    monkeypatch.setattr(llm, "chat", fake_chat)

    c = make_client(tmp_path)
    i = c.post("/api/ideas", json={"title": "T"}).json()["id"]
    nodes = c.post(f"/api/ideas/{i}/hypothesis-map/generate").json()["nodes"]
    hid = nodes[0]["id"]
    assert hid == "H1"  # hierarchical id

    # root plan was created at map-generation time (#4)
    detail = c.get(f"/api/ideas/{i}/hypotheses/{hid}").json()
    assert detail["plan"] and "Datasets" in detail["plan"]
    assert detail["experiment"] is None  # not run yet

    # explicit per-hypothesis plan endpoint also works
    d2 = c.post(f"/api/ideas/{i}/hypotheses/{hid}/plan").json()
    assert "MNIST" in d2["plan"]


def test_hypothesis_edit_tools(tmp_path):
    assert "Added" in H.add(tmp_path, {"statement": "Root"})
    rid = json.loads((tmp_path / ".hypothesis_map.json").read_text())["nodes"][0]["id"]
    H.add(tmp_path, {"statement": "child", "parent_id": rid})
    H.update(tmp_path, {"id": rid, "status": "supported"})
    data = json.loads((tmp_path / ".hypothesis_map.json").read_text())
    assert data["nodes"][0]["status"] == "supported"
    assert len(data["nodes"][0]["children"]) == 1
    H.remove(tmp_path, {"id": rid})
    assert json.loads((tmp_path / ".hypothesis_map.json").read_text())["nodes"] == []


def test_hypothesis_detail_includes_code_and_run_output(tmp_path):
    from paperclaw import service
    from paperclaw.server.store import Store
    store = Store(tmp_path)
    idea = store.add_idea("T")
    store.put_hypothesis_map(idea.id, {"nodes": [
        {"id": "H1", "statement": "x", "status": "supported", "children": []}]})
    hdir = store.idea_path(idea.id) / "hypotheses" / "H1"
    hdir.mkdir(parents=True)
    (hdir / "run.py").write_text("print('hi')")          # phase 1
    (hdir / "stdout.log").write_text("hi")               # run log
    (hdir / "experiment.md").write_text("## results\nok")  # phase 2

    d = service.get_hypothesis_detail(store, idea.id, "H1")
    assert d.code == "print('hi')"
    assert d.experiment == "## results\nok"
    assert d.log == "hi" and d.status == "supported"


def test_run_hypothesis_experiment_simulated(tmp_path, monkeypatch):
    import asyncio
    from paperclaw import service
    from paperclaw.config import LLMSettings
    from paperclaw.server.store import Store

    async def chat(settings, system, messages, max_tokens=2048):
        if "executing the experiment plan" in system:
            return SimpleNamespace(text="```experiment-results\n## E1\n**Status:** POSITIVE\n```", model="x")
        if "TESTING PLAN" in system:
            return SimpleNamespace(text="```plan\n## Feasibility\nFEASIBLE\n```", model="x")
        return SimpleNamespace(text="q", model="x")
    monkeypatch.setattr(llm, "chat", chat)

    store = Store(tmp_path)
    idea = store.add_idea("T")
    store.put_hypothesis_map(idea.id, {"nodes": [
        {"id": "H1", "statement": "x", "status": "untested", "children": []}]})

    d = asyncio.run(service.run_hypothesis_experiment(store, LLMSettings(), idea.id, "H1"))
    assert d.plan          # plan auto-generated when missing
    assert d.experiment and "POSITIVE" in d.experiment


def test_references_generate_route(tmp_path, monkeypatch):
    async def fake_chat(settings, system, messages, max_tokens=2048):
        return SimpleNamespace(text="deep learning\ntransformers", model="x")

    async def fake_search(query, limit=8):
        return [{"title": "A Paper", "authors": ["Jane Doe"], "year": 2021,
                 "venue": "NeurIPS", "doi": "10.1/x"}]

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(literature, "search_recent_papers", fake_search)

    c = make_client(tmp_path)
    i = c.post("/api/ideas", json={"title": "T"}).json()["id"]
    r = c.post(f"/api/ideas/{i}/references/generate")
    assert r.status_code == 200
    entries = r.json()["entries"]
    assert entries and entries[0]["key"] == "doe2021paper"


# ── /generate_report auto-grows the map (verdict status + follow-up hypotheses) ──

def _seed_map(tmp_path):
    """A Store with one untested root H1 and an empty hypotheses/H1/ dir."""
    import asyncio  # noqa: F401 (kept for symmetry with callers)
    from paperclaw.server.store import Store
    store = Store(tmp_path)
    idea = store.add_idea("Idea")
    store.put_hypothesis_map(idea.id, {"ideaId": idea.id, "nodes": [
        {"id": "H1", "statement": "Root H1", "rationale": "core", "status": "untested", "children": []}]})
    hdir = store.idea_path(idea.id) / "hypotheses" / "H1"
    hdir.mkdir(parents=True)
    return store, idea.id, hdir


def test_autoexpand_from_report_sets_status_and_adds_children(tmp_path, monkeypatch):
    import asyncio
    from paperclaw import service
    from paperclaw.config import LLMSettings
    store, idea_id, hdir = _seed_map(tmp_path)
    (hdir / "report.md").write_text(
        "## Verdict\nSUPPORTED\n## Proposed Hypotheses\n- deeper net (sub of H1): test acc\n",
        encoding="utf-8")

    async def fake_chat(settings, system, messages, max_tokens=2048):
        return SimpleNamespace(text='```json\n{"nodes":[{"statement":"child H","test":"t"}]}\n```', model="x")
    monkeypatch.setattr(llm, "chat", fake_chat)

    added = asyncio.run(service.autoexpand_from_report(store, LLMSettings(), idea_id, "H1"))
    assert added == 1
    root = store.get_hypothesis_map(idea_id)["nodes"][0]
    assert root["status"] == "supported"                 # verdict applied deterministically
    assert len(root["children"]) == 1                    # follow-up auto-generated
    assert root["children"][0]["id"] == "H1.1"           # hierarchical id assigned


def test_autoexpand_from_report_noop_without_verdict(tmp_path, monkeypatch):
    import asyncio
    from paperclaw import service
    from paperclaw.config import LLMSettings
    store, idea_id, hdir = _seed_map(tmp_path)
    (hdir / "report.md").write_text("No results yet — run the experiment first.", encoding="utf-8")
    called = []

    async def fake_chat(*a, **k):
        called.append(1)
        return SimpleNamespace(text="", model="x")
    monkeypatch.setattr(llm, "chat", fake_chat)

    added = asyncio.run(service.autoexpand_from_report(store, LLMSettings(), idea_id, "H1"))
    assert added == 0 and not called  # no verdict ⇒ no status change, no expansion call
    assert store.get_hypothesis_map(idea_id)["nodes"][0]["status"] == "untested"


def test_idea_chat_injects_pinned_domain_spec(tmp_path):
    """The idea conversation agent is sandboxed to the idea folder and can't open
    domains/<id>/DOMAIN.md, so the pinned domain's spec is injected into its system
    prompt as read-only reference (giving it the field's literature with authors/year)."""
    from paperclaw import service
    from paperclaw.server.store import Store

    store = Store(tmp_path)
    dom = store.add_domain("Time Series Forecasting")
    store.put_domain_spec(dom.id, "# Time Series Forecasting\n\n## Crucial Papers\n- NsDiff (Smith et al., 2024)\n")
    idea = store.add_idea("TS idea")
    store.put_spec(idea.id, "# TS idea\n\n## Domain & Literature\nPinned to Time Series Forecasting.\n")

    # pinned idea → the domain's DOMAIN.md (incl. authors/year) + its PATH are in the prompt
    sysp = service._idea_chat_system(store, idea.id, store.get_spec(idea.id))
    assert "NsDiff (Smith et al., 2024)" in sysp
    assert "READ-ONLY reference" in sysp
    assert f"domains/{dom.id}/DOMAIN.md" in sysp  # the path is included

    # an unpinned idea gets no domain block
    other = store.add_idea("Unrelated")
    store.put_spec(other.id, "# Unrelated\n\nNo domain here.\n")
    assert "READ-ONLY reference" not in service._idea_chat_system(store, other.id, store.get_spec(other.id))


def test_hypothesis_map_stage_reflects_progress(tmp_path):
    """Map nodes carry a derived progress STAGE (planned/experiment), so an untested
    node that's actually mid-run shows its stage instead of a flat 'untested'."""
    from paperclaw import service
    from paperclaw.server.store import Store

    store = Store(tmp_path)
    idea = store.add_idea("T")
    store.put_hypothesis_map(idea.id, {"nodes": [
        {"id": "H1", "statement": "a", "status": "untested", "children": []},   # plan only
        {"id": "H2", "statement": "b", "status": "untested", "children": []},   # results in
        {"id": "H3", "statement": "c", "status": "supported", "children": []},  # verdicted
        {"id": "H4", "statement": "d", "status": "untested", "children": []},   # nothing
    ]})
    base = store.idea_path(idea.id) / "hypotheses"
    (base / "H1").mkdir(parents=True); (base / "H1" / "plan.md").write_text("plan")
    (base / "H2").mkdir(parents=True); (base / "H2" / "results.json").write_text("{}")

    stages = {n.id: n.stage for n in service.get_hypothesis_map(store, idea.id).nodes}
    assert stages == {"H1": "planned", "H2": "experiment", "H3": "supported", "H4": "untested"}
