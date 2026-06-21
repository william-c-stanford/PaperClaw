"""Tests for the iterative hypothesis-loop pipeline (map-driven, per-hypothesis dirs)."""

import asyncio
from types import SimpleNamespace

import pytest

from paperclaw import iterative_pipeline as ip
from paperclaw import literature, llm
from paperclaw.config import LLMSettings
from paperclaw.server.store import Store

PINNED_SPEC = (
    "# Test idea\n\n"
    "## Domain & Literature (domain pin)\n\n"
    "Machine learning; anchored by Smith et al. 2020.\n\n"
    "## Root Hypotheses\n\nMethod X beats baseline.\n"
)

MAP_JSON = ('```json\n{"nodes":[{"statement":"Root H1","rationale":"core",'
            '"children":[{"statement":"sub","test":"accuracy"}]}]}\n```')


async def fake_chat(settings, system, messages, max_tokens=2048):
    """Non-streaming calls: map generation, reference queries, root plans."""
    if "HYPOTHESIS MAP" in system:
        text = MAP_JSON
    elif "TESTING PLAN" in system:  # HYPOTHESIS_PLAN_SYSTEM (root plans at map-gen)
        text = "```plan\n## Datasets\ntoy\n## Feasibility\nFEASIBLE — fits\n```"
    elif "literature-search queries" in system:
        text = "query one\nquery two"
    else:
        text = ""
    return SimpleNamespace(text=text, model="fake")


def _block_for(system: str, plan_feasible: bool = True) -> str:
    if "TESTING PLAN" in system:  # HYPOTHESIS_PLAN_SYSTEM
        feas = "FEASIBLE" if plan_feasible else "INFEASIBLE"
        return ("```plan\n## Hypothesis\nH1\n## Decision Criteria\n- Supported if: acc>0.8\n"
                f"## Feasibility\n{feas} — fits\n```")
    if "executing the experiment plan" in system:  # EXPERIMENT_SYSTEM
        return "```experiment-results\n## Experiment 1\n**Status:** POSITIVE\n```"
    if "HYPOTHESIS REPORT" in system:  # HYPOTHESIS_REPORT_SYSTEM
        return ("```report\n## Verdict\nSUPPORTED\n## Key Findings\n1. acc 0.9\n"
                "## Enough For Paper\nyes — strong\n```")
    if "figures for a research paper" in system:  # FIGURE_CODE_SYSTEM
        return ('```python\nimport matplotlib; matplotlib.use("Agg")\n'
                'import matplotlib.pyplot as plt\n'
                'plt.figure(); plt.plot([0,1],[0,1]); plt.savefig("fig_method.png")\n'
                'print("saved")\n```')
    if "failed to compile" in system:  # LATEX_FIX_SYSTEM
        return "```latex\n\\documentclass{article}\\begin{document}Fixed.\\end{document}\n```"
    return "```latex\n\\documentclass{article}\\begin{document}Hello world.\\end{document}\n```"


def _make_stream(plan_feasible: bool = True):
    # mirrors llm.stream_chat_thinking → yields typed {"type":"text",...} events
    async def fake_stream(settings, system, messages, max_tokens=2048, **kw):
        yield {"type": "text", "text": _block_for(system, plan_feasible)}
    return fake_stream


async def fake_search(query, limit=8):
    return []  # no network in tests


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch):
    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(llm, "stream_chat_thinking", _make_stream(True))
    monkeypatch.setattr(literature, "search_recent_papers", fake_search)


# ── helpers ───────────────────────────────────────────────────────────────────

def test_parse_enough():
    assert ip._parse_enough("## Enough For Paper\nyes — ok") is True
    assert ip._parse_enough("## Enough For Paper\nno — more") is False


def test_verdict_and_feasibility():
    assert ip._verdict_to_status("## Verdict\nSUPPORTED") == "supported"
    assert ip._verdict_to_status("## Verdict\nPARTIALLY SUPPORTED") == "supported"  # partial = positive node
    assert ip._verdict_to_status("## Verdict\nREFUTED") == "refuted"
    assert ip._verdict_to_status("## Verdict\nINCONCLUSIVE") == "inconclusive"
    assert ip._is_infeasible("## Feasibility\nINFEASIBLE — too big") is True
    assert ip._is_infeasible("## Feasibility\nFEASIBLE — fits") is False


def test_compile_latex_minimal(tmp_path):
    ok, pages, log = ip._compile_latex(tmp_path, "\\documentclass{article}\\begin{document}Hi.\\end{document}")
    assert ok and pages == 1 and (tmp_path / "paper.pdf").is_file()


# A 1×1 PNG — a stand-in for a per-hypothesis result figure.
_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
    "0000000c49444154789c63f8ffff3f0005fe02fe0def46b80000000049454e44ae426082"
)


def test_compile_tex_copies_subdir_figures(tmp_path):
    """A figure under hypotheses/<id>/ must resolve at compile time (the agent
    \\includegraphics it by its workspace-relative path)."""
    (tmp_path / "hypotheses" / "H1").mkdir(parents=True)
    (tmp_path / "hypotheses" / "H1" / "loss.png").write_bytes(_PNG_1x1)
    (tmp_path / "paper.tex").write_text(
        "\\documentclass{article}\\usepackage{graphicx}\\begin{document}"
        "\\includegraphics[width=0.1\\linewidth]{hypotheses/H1/loss.png}"
        "\\end{document}",
        encoding="utf-8")
    ok, pages, log = ip.compile_tex(tmp_path, "paper.tex")
    assert ok, log[-1500:]
    assert (tmp_path / "paper.pdf").is_file()


def test_compile_tex_inputs_subdir_tex_figure(tmp_path):
    """A TikZ/source figure under figures/*.tex must be available to \\input — the
    paper compiles, and the figure can ALSO be compiled standalone by its subdir name
    (regression: that used to crash with FileNotFoundError on the temp dest dir)."""
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "pareto.tex").write_text(
        "\\framebox{Pareto figure}", encoding="utf-8")
    (tmp_path / "paper.tex").write_text(
        "\\documentclass{article}\\begin{document}\\input{figures/pareto.tex}\\end{document}",
        encoding="utf-8")
    # the paper \input's the subdir .tex figure source
    ok, _, log = ip.compile_tex(tmp_path, "paper.tex")
    assert ok, log[-1500:]

    # compiling the figure standalone by its SUBDIR name no longer crashes; PDF lands
    # next to the source (figures/pareto.pdf), not the workspace root
    (tmp_path / "figures" / "fig.tex").write_text(
        "\\documentclass{standalone}\\begin{document}\\framebox{X}\\end{document}",
        encoding="utf-8")
    ok2, _, log2 = ip.compile_tex(tmp_path, "figures/fig.tex")
    assert ok2, log2[-1500:]
    assert (tmp_path / "figures" / "fig.pdf").is_file()


# ── full loop ─────────────────────────────────────────────────────────────────

def _run(store, idea_id, **kw):
    events = []

    async def go():
        async for ev in ip.stream_iterative_research_events(store, LLMSettings(), idea_id, **kw):
            events.append(ev)
    asyncio.run(go())
    return events


def test_domain_optional_no_block(tmp_path):
    """A domain is OPTIONAL — an idea without one is NOT blocked; the run proceeds with
    the domain context left blank (no 'needs_domain' error)."""
    store = Store(tmp_path)
    idea = store.add_idea("Unpinned")  # default spec has no pinned domain
    events = _run(store, idea.id)
    assert not any(e["type"] == "needs_domain" for e in events)        # no hard block
    assert any("no domain linked" in e.get("text", "") for e in events)  # proceeds, blank


def test_iterative_per_hypothesis_dirs_and_paper(tmp_path):
    store = Store(tmp_path)
    idea = store.add_idea("Test idea")
    store.put_spec(idea.id, PINNED_SPEC)

    events = _run(store, idea.id, max_hypotheses=1)
    types = [e["type"] for e in events]
    assert "round" in types and "round_done" in types
    assert any(e["type"] == "page_check" and e["compliant"] for e in events)
    assert "paper_ready" in types and "done" in types

    # the map root got its own directory with plan/experiment/report (#6)
    hmap = store.get_hypothesis_map(idea.id)
    hid = hmap["nodes"][0]["id"]
    hdir = store.idea_path(idea.id) / "hypotheses" / hid
    assert (hdir / "plan.md").is_file()
    assert (hdir / "experiment.md").is_file()
    assert (hdir / "report.md").is_file()
    assert (store.idea_path(idea.id) / "paper.pdf").is_file()
    # node status updated from the report verdict
    assert hmap["nodes"][0]["status"] == "supported"


def test_feasibility_gate_blocks(tmp_path, monkeypatch):
    # Root plans are created at map-gen via llm.chat — make that plan INFEASIBLE.
    async def chat(settings, system, messages, max_tokens=2048):
        if "HYPOTHESIS MAP" in system:
            return SimpleNamespace(text=MAP_JSON, model="x")
        if "TESTING PLAN" in system:
            return SimpleNamespace(text="```plan\n## Feasibility\nINFEASIBLE — needs 1000 GPUs\n```", model="x")
        return SimpleNamespace(text="q", model="x")
    monkeypatch.setattr(llm, "chat", chat)
    store = Store(tmp_path)
    idea = store.add_idea("Test idea")
    store.put_spec(idea.id, PINNED_SPEC)

    events = _run(store, idea.id, max_hypotheses=1)
    assert any(e.get("type") == "hypothesis_status" and e["status"] == "blocked" for e in events)

    hmap = store.get_hypothesis_map(idea.id)
    hid = hmap["nodes"][0]["id"]
    hdir = store.idea_path(idea.id) / "hypotheses" / hid
    assert (hdir / "plan.md").is_file()
    assert not (hdir / "experiment.md").is_file()  # blocked before experiment
    assert hmap["nodes"][0]["status"] == "blocked"


def test_iterative_generates_paper_figures(tmp_path):
    store = Store(tmp_path)
    idea = store.add_idea("Test idea")
    store.put_spec(idea.id, PINNED_SPEC)
    _run(store, idea.id, max_hypotheses=1)
    # conceptual figure drawn into figures/ and available to the paper
    assert (store.idea_path(idea.id) / "figures" / "fig_method.png").is_file()


def test_target_positive_stops_after_n_supported(tmp_path, monkeypatch):
    """Auto mode: with target_positive=2 the loop stops once 2 hypotheses are
    SUPPORTED, leaving later roots untested (even under a larger max budget)."""
    store = Store(tmp_path)
    idea = store.add_idea("Idea")
    store.put_spec(idea.id, PINNED_SPEC)
    three_roots = ('```json\n{"nodes":[{"statement":"H one","rationale":"a"},'
                   '{"statement":"H two","rationale":"b"},'
                   '{"statement":"H three","rationale":"c"}]}\n```')

    async def chat3(settings, system, messages, max_tokens=2048):
        if "HYPOTHESIS MAP" in system:
            text = three_roots
        elif "TESTING PLAN" in system:
            text = "```plan\n## Feasibility\nFEASIBLE — fits\n```"
        elif "literature-search queries" in system:
            text = "q"
        else:
            text = ""  # expansion → no children, keep the tree bounded
        return SimpleNamespace(text=text, model="x")
    monkeypatch.setattr(llm, "chat", chat3)  # reports stream SUPPORTED via the autouse fixture

    events = []

    async def go():
        async for ev in ip.stream_iterative_research_events(
            store, LLMSettings(), idea.id, max_hypotheses=5, page_limit=8, target_positive=2):
            events.append(ev)
    asyncio.run(go())

    assert len([e for e in events if e["type"] == "round"]) == 2  # stopped at 2, not 3/5
    statuses = [n.get("status", "untested") for n in store.get_hypothesis_map(idea.id)["nodes"]]
    assert statuses.count("supported") == 2 and "untested" in statuses


def test_count_supported_across_tree():
    """Counts SUPPORTED nodes at every depth (drives the resume stop condition)."""
    m = {"nodes": [
        {"id": "H1", "status": "supported", "children": [
            {"id": "H1.1", "status": "refuted"},
            {"id": "H1.2", "status": "supported"}]},
        {"id": "H2", "status": "untested"},
    ]}
    assert ip._count_supported(m) == 2


def test_resume_counts_prior_positives_and_stops(tmp_path, monkeypatch):
    """RESUME: a map already holding target_positive SUPPORTED nodes must NOT test more
    — it counts prior-run positives, emits them, and goes straight to the paper."""
    store = Store(tmp_path)
    idea = store.add_idea("Idea")
    store.put_spec(idea.id, PINNED_SPEC)
    # a prior run left 2 SUPPORTED roots + 1 still untested
    store.put_hypothesis_map(idea.id, {"nodes": [
        {"id": "H1", "statement": "H one", "status": "supported"},
        {"id": "H2", "statement": "H two", "status": "supported"},
        {"id": "H3", "statement": "H three", "status": "untested"},
    ]})

    async def chat(settings, system, messages, max_tokens=2048):
        return SimpleNamespace(text="", model="x")
    monkeypatch.setattr(llm, "chat", chat)

    events = []

    async def go():
        async for ev in ip.stream_iterative_research_events(
            store, LLMSettings(), idea.id, max_hypotheses=5, page_limit=8, target_positive=2):
            events.append(ev)
    asyncio.run(go())

    assert not [e for e in events if e["type"] == "round"]  # tested NOTHING more
    pos = [e for e in events if e["type"] == "positives"]
    assert pos and pos[0]["positives"] == 2                 # surfaced the prior positives
    assert store.get_hypothesis_map(idea.id)["nodes"][2]["status"] == "untested"  # H3 left alone


def test_collect_paper_rounds_includes_prior_session_reports(tmp_path):
    """RESUME: the paper is rebuilt from ON-DISK artifacts, so hypotheses proven in an
    EARLIER session (with no in-session round) are still included — not an empty paper."""
    store = Store(tmp_path)
    idea = store.add_idea("Idea")
    store.put_spec(idea.id, PINNED_SPEC)
    store.put_hypothesis_map(idea.id, {"nodes": [
        {"id": "H1", "statement": "H one", "status": "supported"},
        {"id": "H2", "statement": "H two", "status": "supported"},
        {"id": "H3", "statement": "H three", "status": "untested"},  # never tested
    ]})
    idea_path = store.idea_path(idea.id)
    for hid in ("H1", "H2"):
        hd = idea_path / "hypotheses" / hid
        hd.mkdir(parents=True)
        (hd / "plan.md").write_text(f"plan {hid}", encoding="utf-8")
        (hd / "experiment.md").write_text(f"results {hid}", encoding="utf-8")
        (hd / "report.md").write_text(f"## Verdict\nSUPPORTED\nreport {hid}", encoding="utf-8")

    rounds = ip._collect_paper_rounds(idea_path, store, idea.id, [])  # empty session (resume)
    assert [r["id"] for r in rounds] == ["H1", "H2"]   # H3 (untested) excluded; priors included
    assert rounds[0]["report"].endswith("report H1")
    assert rounds[1]["experiment"] == "results H2"
    assert [r["k"] for r in rounds] == [1, 2]          # renumbered sequentially


def test_agent_command_available():
    from paperclaw.agents import agent_command_available
    from paperclaw.server.models import RunConfig
    assert agent_command_available(RunConfig(experimentMode="cli", agentCommand="echo hi {prompt}")) is True
    assert agent_command_available(RunConfig(experimentMode="cli", agentCommand="__no_such_bin__ {prompt}")) is False
    assert agent_command_available(RunConfig(experimentMode="cli", agentCommand="")) is False


def test_cli_falls_back_to_in_process_when_unavailable(tmp_path, monkeypatch):
    """cli mode with an uninstalled CLI binary → run the in-process agent instead
    (prefixed with a status note), so the experiment still runs."""
    from paperclaw import agents
    from paperclaw.server.models import RunConfig
    monkeypatch.setattr(agents, "agent_command_available", lambda rc: False)

    async def fake_exec(settings, idea_ctx, plan, out_dir, run_cfg):
        yield {"type": "result", "result": {"markdown": "ran in-process"}}
    monkeypatch.setattr(agents, "run_agentic_experiment", fake_exec)

    rc = RunConfig(experimentMode="cli", agentCommand="claude -p {prompt}")
    runner = ip._select_experiment_runner(None, "ctx", "plan", tmp_path, rc, None)
    evs = []

    async def go():
        async for ev in runner:
            evs.append(ev)
    asyncio.run(go())
    assert evs[0]["type"] == "status" and "in-process" in evs[0]["text"]
    assert evs[-1]["result"]["markdown"] == "ran in-process"


def test_ssh_mode_without_target_is_graceful(tmp_path):
    from paperclaw.server.models import RunConfig
    store = Store(tmp_path)
    store.save_run_config(RunConfig(experimentMode="ssh"))  # no SSH remotes configured
    idea = store.add_idea("Test idea")
    store.put_spec(idea.id, PINNED_SPEC)

    events = _run(store, idea.id, max_hypotheses=1)
    # pipeline must still complete (no crash); experiment notes the missing remote
    assert any(e["type"] == "done" for e in events)
    hid = store.get_hypothesis_map(idea.id)["nodes"][0]["id"]
    exp = (store.idea_path(idea.id) / "hypotheses" / hid / "experiment.md").read_text()
    assert "SSH" in exp


def test_report_expands_tree_and_tests_child(tmp_path, monkeypatch):
    async def chat(settings, system, messages, max_tokens=2048):
        if "HYPOTHESIS MAP" in system:
            return SimpleNamespace(text=MAP_JSON, model="x")
        if "TESTING PLAN" in system:
            return SimpleNamespace(text="```plan\n## Feasibility\nFEASIBLE — fits\n```", model="x")
        if "worth testing next" in system:  # HYPOTHESIS_EXPAND_SYSTEM
            return SimpleNamespace(text='```json\n{"nodes":[{"statement":"child H","test":"t"}]}\n```', model="x")
        return SimpleNamespace(text="q", model="x")

    async def stream(settings, system, messages, max_tokens=2048, **kw):
        if "TESTING PLAN" in system:
            text = "```plan\n## Hypothesis\nH\n## Feasibility\nFEASIBLE — fits\n```"
        elif "executing the experiment plan" in system:
            text = "```experiment-results\n## E\n**Status:** POSITIVE\n```"
        elif "HYPOTHESIS REPORT" in system:
            text = "```report\n## Verdict\nSUPPORTED\n## Enough For Paper\nno — need more\n```"
        elif "figures for a research paper" in system:
            text = "```python\nprint(1)\n```"
        else:
            text = "```latex\n\\documentclass{article}\\begin{document}x\\end{document}\n```"
        yield {"type": "text", "text": text}

    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "stream_chat_thinking", stream)
    monkeypatch.setattr(literature, "search_recent_papers", fake_search)

    store = Store(tmp_path)
    idea = store.add_idea("Test idea")
    store.put_spec(idea.id, PINNED_SPEC)
    events = _run(store, idea.id, max_hypotheses=2)

    assert any(e["type"] == "hypothesis_expanded" for e in events)
    root = store.get_hypothesis_map(idea.id)["nodes"][0]
    assert root.get("children")  # child sub-hypothesis was added
    child_id = root["children"][0]["id"]
    # with budget 2, the added child was also tested (has its own report)
    assert (store.idea_path(idea.id) / "hypotheses" / child_id / "report.md").is_file()


def test_stream_hypothesis_experiment_feedback(tmp_path):
    from paperclaw import service
    from paperclaw.config import LLMSettings
    store = Store(tmp_path)
    idea = store.add_idea("Test idea")
    store.put_spec(idea.id, PINNED_SPEC)
    store.put_hypothesis_map(idea.id, {"nodes": [
        {"id": "H1", "statement": "x", "status": "untested", "children": []}]})

    events = []

    async def go():
        async for ev in service.stream_hypothesis_experiment(store, LLMSettings(), idea.id, "H1"):
            events.append(ev)
    asyncio.run(go())

    types = [e["type"] for e in events]
    assert "phase" in types and "delta" in types and "done" in types  # full feedback streamed
    assert (store.idea_path(idea.id) / "hypotheses" / "H1" / "experiment.md").is_file()


def test_ocr_check_graceful(tmp_path):
    ok, _, _ = ip._compile_latex(tmp_path, "\\documentclass{article}\\begin{document}Hi.\\end{document}")
    assert ok
    # tesseract/pdftoppm not installed in this env → graceful None (no crash)
    assert ip._ocr_check(tmp_path / "paper.pdf") is None


def test_page_limit_enforcement(tmp_path, monkeypatch):
    async def chat(settings, system, messages, max_tokens=2048):
        if "HYPOTHESIS MAP" in system:
            return SimpleNamespace(text=MAP_JSON, model="x")
        return SimpleNamespace(text="q", model="x")

    async def stream(settings, system, messages, max_tokens=2048, **kw):
        if "TESTING PLAN" in system:
            text = "```plan\n## Feasibility\nFEASIBLE\n```"
        elif "executing the experiment plan" in system:
            text = "```experiment-results\n## E\n**Status:** POSITIVE\n```"
        elif "HYPOTHESIS REPORT" in system:
            text = "```report\n## Verdict\nSUPPORTED\n## Enough For Paper\nyes\n```"
        elif "figures for a research paper" in system:
            text = "```python\nprint(1)\n```"
        elif "shortened" in system:  # LATEX_SHORTEN_SYSTEM → 1 page
            text = "```latex\n\\documentclass{article}\\begin{document}Short.\\end{document}\n```"
        else:  # LATEX_PAPER_SYSTEM → 3 pages (over the limit)
            text = "```latex\n\\documentclass{article}\\begin{document}A\\newpage B\\newpage C\\end{document}\n```"
        yield {"type": "text", "text": text}

    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "stream_chat_thinking", stream)
    monkeypatch.setattr(literature, "search_recent_papers", fake_search)
    store = Store(tmp_path)
    idea = store.add_idea("Test idea")
    store.put_spec(idea.id, PINNED_SPEC)

    events = _run(store, idea.id, max_hypotheses=1, page_limit=1)
    page_checks = [e for e in events if e["type"] == "page_check"]
    assert page_checks and page_checks[-1]["pages"] == 1 and page_checks[-1]["compliant"]


def test_hypothesis_detail(tmp_path):
    from paperclaw import service
    store = Store(tmp_path)
    idea = store.add_idea("Test idea")
    store.put_spec(idea.id, PINNED_SPEC)
    _run(store, idea.id, max_hypotheses=1)

    hid = store.get_hypothesis_map(idea.id)["nodes"][0]["id"]
    d = service.get_hypothesis_detail(store, idea.id, hid)
    assert d.status == "supported"
    assert d.plan and d.experiment and d.report
