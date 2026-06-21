"""Autonomous research pipeline for PaperClaw.

Four sequential LLM phases take an idea from spec → paper:
  1. plan       — design research questions, hypotheses, and experiments
  2. experiment — simulate experiment execution, produce realistic results
  3. analysis   — synthesize findings (positive + negative → future work)
  4. paper      — write a complete academic paper in Markdown

Each completed phase's output is saved to the idea folder as a hidden file
(`.research_plan.md`, etc., plus `paper.md` / `paper.pdf`). A run is therefore
RESUMABLE: re-invoking the pipeline skips phases whose artifacts already exist
(unless `restart=True`), so a stopped run continues from where it left off.
(Flat files, not a subdirectory — the store deletes ideas by unlinking files.)

Events yielded:
  {"type": "phase",      "phase": str, "label": str}       — phase starting
  {"type": "delta",      "text": str}                       — streaming LLM chunk
  {"type": "phase_done", "phase": str, "content": str}      — phase complete
  {"type": "spec_updated"}                                   — IDEA.md updated
  {"type": "paper_ready", "download_url": str}               — paper saved
  {"type": "done"}
  {"type": "error",      "message": str}
"""

import re
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import AsyncIterator

from paperclaw import experiments, llm
from paperclaw.config import LLMSettings
from paperclaw.prompts.pipeline import (
    ANALYSIS_SYSTEM,
    EXPERIMENT_SYSTEM,
    PAPER_TEMPLATE,
    PLAN_SYSTEM,
)
from paperclaw.server.store import Store

# Hidden artifact filenames in the idea folder, used to resume interrupted runs.
_ARTIFACTS = {
    "plan": ".research_plan.md",
    "experiment": ".research_experiments.md",
    "analysis": ".research_analysis.md",
}
_PAPER_FILES = ("paper.md", "paper.pdf")

# A full academic paper (Abstract → Conclusion → References, with tables/math)
# easily exceeds a few thousand tokens. 4096 truncated papers mid-section
# (~page 8). The paper phase streams, so a large cap is safe — streaming avoids
# the SDK's non-streaming HTTP-timeout guard. claude-opus-4-8 supports up to
# 128K output tokens; 16K is plenty for a complete paper without risking
# rejection from OpenAI-compatible endpoints that cap lower.
_PAPER_MAX_TOKENS = 16000

# ── Parsing helpers ───────────────────────────────────────────────────────────

_BLOCK_RE = re.compile(r"```([\w-]+)\s*\n(.*?)```", re.DOTALL)

_CURRENT_FINDINGS_RE = re.compile(
    r"## Current Findings \(for IDEA\.md\)\n(.*?)(?=\n## |\Z)", re.DOTALL
)

_FENCE_STRIP_RE = re.compile(r"^```[\w-]*\s*\n?|```\s*$", re.MULTILINE)


def _extract_block(text: str, tag: str) -> str:
    """Extract content from a fenced block. Case-insensitive tag match.
    Falls back to stripping any opening/closing fences from the raw text."""
    for m in _BLOCK_RE.finditer(text):
        if m.group(1).lower() == tag.lower():
            return m.group(2).strip()
    # Fallback: strip any fence markers (LLM used wrong tag or no tag)
    stripped = _FENCE_STRIP_RE.sub("", text).strip()
    return stripped if stripped else text.strip()


_PAPER_OPEN_RE = re.compile(r"```paper[^\n]*\n", re.IGNORECASE)
_PAPER_CLOSE_RE = re.compile(r"\n```\s*$")


def _extract_paper(text: str) -> str:
    """Extract the paper body from a ```paper fenced block.

    A paper routinely contains its OWN nested ``` code fences (pseudocode,
    algorithms in Methodology), so the generic first-closing-fence matcher in
    :func:`_extract_block` truncates the paper at the first inner fence. Instead,
    take everything from after the ```paper opening fence to the end and drop a
    single trailing closing fence if present — this keeps nested fences intact
    and still salvages a paper whose closing fence was lost to truncation.
    """
    m = _PAPER_OPEN_RE.search(text)
    if not m:
        return _extract_block(text, "paper")  # no opening fence — best effort
    body = text[m.end():]
    body = _PAPER_CLOSE_RE.sub("", body.rstrip())  # remove the paper's own closing fence
    return body.strip()


_DOMAIN_PIN_RE = re.compile(
    r"## Domain & Literature.*?\n(.*?)(?=\n## |\Z)", re.DOTALL
)


def domain_is_pinned(spec: str) -> bool:
    """True if the IDEA.md has a real pinned domain (not the TBD placeholder).

    The research pipeline requires a pinned domain so the paper stays grounded
    in the idea's actual field rather than drifting into an unrelated one.
    """
    m = _DOMAIN_PIN_RE.search(spec)
    if not m:
        return False
    body = re.sub(r"<!--.*?-->", "", m.group(1), flags=re.DOTALL).strip()
    if not body:
        return False
    # A bare placeholder body ("TBD", "_TBD_", "N/A", "Not pinned yet") means "not pinned
    # yet". Only treat it as unpinned when the WHOLE pin is a SHORT placeholder — NOT when
    # a real, substantive pin merely mentions e.g. "(authors/year TBD)" about one paper
    # (a long pin like that is genuinely pinned, just with a citation detail to fill).
    # Strip markdown emphasis first so `_TBD_` still matches the word boundary.
    low = re.sub(r"[*_`]", "", body).strip().lower()
    if len(low) <= 40 and re.search(r"\b(tbd|n/?a|none|not pinned|to be determined)\b", low):
        return False
    return True


def _update_spec_findings(spec: str, new_content: str) -> str:
    """Replace the '## Current Findings' section in IDEA.md."""
    new_content = new_content.strip()
    pattern = re.compile(r"## Current Findings\n(.*?)(?=\n## |\Z)", re.DOTALL)
    updated, n = pattern.subn(f"## Current Findings\n{new_content}\n", spec, count=1)
    if n == 0:
        updated = spec.rstrip() + f"\n\n## Current Findings\n{new_content}\n"
    return updated


# ── Artifact persistence (for resume) ─────────────────────────────────────────

def _artifact_path(idea_path: Path | None, phase: str) -> Path | None:
    if idea_path is None:
        return None
    return idea_path / _ARTIFACTS[phase]


def _load_artifact(idea_path: Path | None, phase: str) -> str | None:
    p = _artifact_path(idea_path, phase)
    if p is not None and p.is_file():
        return p.read_text(encoding="utf-8")
    return None


def _save_artifact(idea_path: Path | None, phase: str, content: str) -> None:
    p = _artifact_path(idea_path, phase)
    if p is not None:
        p.write_text(content, encoding="utf-8")


def _clear_artifacts(idea_path: Path | None) -> None:
    if idea_path is None:
        return
    for name in (*_ARTIFACTS.values(), *_PAPER_FILES):
        f = idea_path / name
        if f.exists():
            f.unlink()
    exp_dir = idea_path / "experiments"
    if exp_dir.is_dir():
        shutil.rmtree(exp_dir)  # executed-run code/results/figures


def save_phase_partial(idea_path: Path | None, phase: str, raw_content: str) -> bool:
    """Persist partial streaming content for *phase* as a resumable artifact.

    Strips fenced-block markers (the LLM might not have closed them yet).
    Returns True if anything was saved, False if content was empty or phase unknown.
    """
    if idea_path is None or phase not in _ARTIFACTS:
        return False
    content = _FENCE_STRIP_RE.sub("", raw_content).strip()
    if not content:
        return False
    _save_artifact(idea_path, phase, content)
    return True


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def stream_auto_research_events(
    store: Store,
    settings: LLMSettings,
    idea_id: str,
    restart: bool = False,
) -> AsyncIterator[dict]:
    """Yield SSE event dicts driving the autonomous research pipeline.

    Resumable: phases whose artifacts already exist on disk are replayed
    instantly (skipping the LLM call) unless ``restart`` is True.
    """

    spec = store.get_spec(idea_id)
    if spec is None:
        yield {"type": "error", "message": "Idea not found"}
        return

    # Gate: refuse to run without a pinned domain. An unpinned idea has no field
    # to anchor the paper, which is exactly how a paper drifts into the wrong
    # domain. Tell the user to pin one first.
    if not domain_is_pinned(spec):
        yield {
            "type": "needs_domain",
            "message": (
                "This idea has no pinned domain, so research can't start — the "
                "paper would have no field to stay grounded in. Pin a domain "
                "first: run /create_domain (or use the Domains wizard) and fill "
                "the idea's “Domain & Literature (domain pin)” section, then try "
                "Auto again."
            ),
        }
        return

    idea_path = store.idea_path(idea_id)
    if restart:
        _clear_artifacts(idea_path)

    # The idea's own IDEA.md is the single source of truth — it already carries
    # the pinned domain and literature. We deliberately do NOT inject the
    # globally-selected sidebar domains here: doing so let an unrelated selected
    # domain (e.g. "Time Series Generation") bleed into and override an idea
    # about a different field. The paper must follow the idea, nothing else.
    base_ctx = f"IDEA.md:\n{spec}"

    paper_system = PAPER_TEMPLATE.replace("{today}", date.today().strftime("%B %d, %Y"))

    # ── Phase 1: Research Plan ────────────────────────────────────────────────
    saved = None if restart else _load_artifact(idea_path, "plan")
    if saved is not None:
        yield {"type": "phase", "phase": "plan", "label": "Loaded saved research plan"}
        plan_content = saved
    else:
        yield {"type": "phase", "phase": "plan", "label": "Designing research plan..."}
        plan_raw = ""
        try:
            async for chunk in llm.stream_chat(
                settings, PLAN_SYSTEM,
                [{"role": "user", "content": base_ctx}], max_tokens=2048,
            ):
                plan_raw += chunk
                yield {"type": "delta", "text": chunk}
        except (llm.LLMNotConfigured, llm.LLMError) as exc:
            yield {"type": "error", "message": str(exc)}
            return
        plan_content = _extract_block(plan_raw, "research-plan")
        _save_artifact(idea_path, "plan", plan_content)
    yield {"type": "phase_done", "phase": "plan", "content": plan_content}

    # ── Phase 2: Experiments ──────────────────────────────────────────────────
    saved = None if restart else _load_artifact(idea_path, "experiment")
    if saved is not None:
        yield {"type": "phase", "phase": "experiment", "label": "Loaded saved experiments"}
        exp_content = saved
    elif store.get_run_config().experiment_mode == "executed":
        # Executed mode: generate + run real code on the local host. The agentic
        # generate→run→debug loop streams code and execution logs as deltas; the
        # terminal result carries the real experiment-results markdown.
        run_cfg = store.get_run_config()
        yield {"type": "phase", "phase": "experiment", "label": "Running experiments (executing code)…"}
        exp_dir = (idea_path or Path(tempfile.mkdtemp())) / "experiments"
        result: dict | None = None
        try:
            async for ev in experiments.run_local_code(
                settings, base_ctx, plan_content, exp_dir, run_cfg,
            ):
                if ev["type"] in ("delta", "status"):
                    yield {"type": "delta", "text": ev["text"]}
                elif ev["type"] == "result":
                    result = ev["result"]
        except Exception as exc:  # never let a runner bug kill the whole pipeline
            yield {"type": "delta", "text": f"\n[execution error: {exc}]\n"}
        exp_content = (result or {}).get("markdown") or "_No experiment results produced._"
        _save_artifact(idea_path, "experiment", exp_content)
    else:
        yield {"type": "phase", "phase": "experiment", "label": "Running experiments..."}
        exp_ctx = f"{base_ctx}\n\nResearch Plan:\n{plan_content}"
        exp_raw = ""
        try:
            async for chunk in llm.stream_chat(
                settings, EXPERIMENT_SYSTEM,
                [{"role": "user", "content": exp_ctx}], max_tokens=2048,
            ):
                exp_raw += chunk
                yield {"type": "delta", "text": chunk}
        except (llm.LLMNotConfigured, llm.LLMError) as exc:
            yield {"type": "error", "message": str(exc)}
            return
        exp_content = _extract_block(exp_raw, "experiment-results")
        _save_artifact(idea_path, "experiment", exp_content)
    yield {"type": "phase_done", "phase": "experiment", "content": exp_content}

    # ── Phase 3: Analysis ─────────────────────────────────────────────────────
    saved = None if restart else _load_artifact(idea_path, "analysis")
    if saved is not None:
        yield {"type": "phase", "phase": "analysis", "label": "Loaded saved analysis"}
        analysis_content = saved
    else:
        yield {"type": "phase", "phase": "analysis", "label": "Analyzing findings..."}
        analysis_ctx = (
            f"{base_ctx}\n\nResearch Plan:\n{plan_content}"
            f"\n\nExperiment Results:\n{exp_content}"
        )
        analysis_raw = ""
        try:
            async for chunk in llm.stream_chat(
                settings, ANALYSIS_SYSTEM,
                [{"role": "user", "content": analysis_ctx}], max_tokens=2048,
            ):
                analysis_raw += chunk
                yield {"type": "delta", "text": chunk}
        except (llm.LLMNotConfigured, llm.LLMError) as exc:
            yield {"type": "error", "message": str(exc)}
            return
        analysis_content = _extract_block(analysis_raw, "findings")
        _save_artifact(idea_path, "analysis", analysis_content)

        # Update IDEA.md Current Findings (only on a fresh analysis, not on resume)
        m = _CURRENT_FINDINGS_RE.search(analysis_content)
        if m:
            store.put_spec(idea_id, _update_spec_findings(spec, m.group(1)))
            yield {"type": "spec_updated"}
    yield {"type": "phase_done", "phase": "analysis", "content": analysis_content}

    # ── Phase 4: Write Paper ──────────────────────────────────────────────────
    existing_paper = None
    if not restart and idea_path is not None and (idea_path / "paper.md").is_file():
        existing_paper = (idea_path / "paper.md").read_text(encoding="utf-8")

    if existing_paper is not None:
        yield {"type": "phase", "phase": "paper", "label": "Loaded saved paper"}
        paper_content = existing_paper
    else:
        yield {"type": "phase", "phase": "paper", "label": "Writing paper..."}
        paper_ctx = (
            f"{base_ctx}\n\nResearch Plan:\n{plan_content}"
            f"\n\nExperiment Results:\n{exp_content}"
            f"\n\nFindings:\n{analysis_content}"
        )
        paper_raw = ""
        try:
            async for chunk in llm.stream_chat(
                settings, paper_system,
                [{"role": "user", "content": paper_ctx}], max_tokens=_PAPER_MAX_TOKENS,
            ):
                paper_raw += chunk
                yield {"type": "delta", "text": chunk}
        except (llm.LLMNotConfigured, llm.LLMError) as exc:
            yield {"type": "error", "message": str(exc)}
            return
        paper_content = _extract_paper(paper_raw)
        if idea_path is not None:
            (idea_path / "paper.md").write_text(paper_content, encoding="utf-8")
            try:
                from paperclaw.paper import markdown_to_pdf
                markdown_to_pdf(paper_content, idea_path / "paper.pdf")
            except Exception:
                pass  # PDF failed; Markdown file is still served by /paper route

    yield {"type": "phase_done", "phase": "paper", "content": paper_content}
    yield {"type": "paper_ready", "download_url": f"/api/ideas/{idea_id}/paper"}
    yield {"type": "done"}
