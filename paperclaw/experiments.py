"""Pluggable experiment execution.

Two modes (selected by :class:`RunConfig.experiment_mode`):

  * ``simulated`` — the original behavior: the LLM narrates plausible results.
    Driven inline by ``research_pipeline.py`` (``EXPERIMENT_SYSTEM``). Preserved
    so the classic pipeline always works.

  * ``executed`` — :func:`run_local_code` generates a real Python script, runs it
    as a subprocess on the local host (with a wall-clock timeout), and feeds any
    traceback back to the model to fix and retry. Real measured numbers and
    figures replace the simulated narrative.

SECURITY: ``executed`` mode runs arbitrary model-written code as a subprocess
with the backend's permissions (no container). This is intended for a
self-hosted, trusted deployment — the user deploys their own backend. The only
guards are a timeout and a dedicated working directory.

:func:`run_local_code` is an async generator of event dicts so the pipeline can
stream live progress over SSE:

  {"type": "delta",  "text": str}    — code streaming from the model
  {"type": "status", "text": str}    — execution progress line
  {"type": "result", "result": dict} — terminal; the experiment-results markdown
                                        + provenance/figures/attempts/error
"""

import asyncio
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import AsyncIterator

from paperclaw import llm
from paperclaw.config import LLMSettings
from paperclaw.prompts.pipeline import (
    EXPERIMENT_CODE_FIX,
    EXPERIMENT_CODE_SYSTEM,
    FIGURE_CODE_SYSTEM,
)
from paperclaw.server.models import RunConfig

_PY_BLOCK = re.compile(r"```python\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_STDERR_TAIL = 4000  # chars of error fed back to the model on a failed run


def _extract_code(text: str) -> str:
    m = _PY_BLOCK.search(text)
    return (m.group(1) if m else text).strip()


def _exec(python_exe: str, out_dir: Path, script: str, timeout: int) -> tuple[int, str, str]:
    """Run *script* as a subprocess in out_dir; return (returncode, stdout, stderr).
    ``timeout <= 0`` means NO timeout (experiments may run for hours)."""
    to = timeout if timeout and timeout > 0 else None
    try:
        p = subprocess.run(
            [python_exe, script],
            cwd=str(out_dir), capture_output=True, text=True, timeout=to,
        )
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = (exc.stderr or "") + f"\n[TIMEOUT after {timeout}s]"
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return 124, out, err


def _ssh_base(target, scp: bool = False) -> list[str]:
    """Common ssh/scp option list for a target (key-based, non-interactive)."""
    cmd = ["scp" if scp else "ssh",
           "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           "-o", "StrictHostKeyChecking=accept-new",
           ("-P" if scp else "-p"), str(target.port)]
    if target.key_path:
        cmd += ["-i", target.key_path]
    return cmd


def _exec_remote(target, run_config: RunConfig, out_dir: Path) -> tuple[int, str, str]:
    """Push run.py to a fresh remote temp dir, run it, pull results.json + PNGs back."""
    label = f"{target.user}@{target.host}"
    py = run_config.python_path or "python3"
    remote = f"/tmp/paperclaw_{uuid.uuid4().hex[:10]}"
    ssh, scp = _ssh_base(target), _ssh_base(target, scp=True)
    run_to = None  # experiments run with NO wall-clock timeout (they can take hours)
    try:
        subprocess.run(ssh + [label, f"mkdir -p {remote}"], capture_output=True, text=True, timeout=30)
        push = subprocess.run(scp + [str(out_dir / "run.py"), f"{label}:{remote}/run.py"],
                              capture_output=True, text=True, timeout=120)
        if push.returncode != 0:
            return 1, "", f"scp push failed: {push.stderr or push.stdout}"
        run = subprocess.run(ssh + [label, f"cd {remote} && {py} run.py"],
                             capture_output=True, text=True, timeout=run_to)
        # pull artifacts back (best-effort — they may not exist)
        subprocess.run(scp + [f"{label}:{remote}/results.json", f"{out_dir}/"],
                       capture_output=True, text=True, timeout=60)
        subprocess.run(scp + [f"{label}:{remote}/*.png", f"{out_dir}/"],
                       capture_output=True, text=True, timeout=120)
        subprocess.run(ssh + [label, f"rm -rf {remote}"], capture_output=True, text=True, timeout=30)
        return run.returncode, run.stdout or "", run.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", f"remote run timed out after {timeout}s"
    except Exception as exc:  # ssh/scp missing, host unreachable
        return 1, "", f"ssh error: {exc}"


def _load_results(out_dir: Path) -> dict | None:
    p = out_dir / "results.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _metrics_table(metrics: dict) -> list[str]:
    """Render a {method: {metric: value}} dict as a Markdown table."""
    methods = list(metrics.keys())
    metric_names: list[str] = []
    for m in methods:
        if isinstance(metrics[m], dict):
            for k in metrics[m]:
                if k not in metric_names:
                    metric_names.append(k)
    if not methods or not metric_names:
        return []
    lines = ["| Metric | " + " | ".join(methods) + " |",
             "|" + "---|" * (len(methods) + 1)]
    for name in metric_names:
        row = [name]
        for m in methods:
            v = metrics[m].get(name) if isinstance(metrics[m], dict) else None
            row.append("—" if v is None else (f"{v:.4g}" if isinstance(v, (int, float)) else str(v)))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _render_results_md(results: dict | None, stdout: str, figures: list[str]) -> str:
    """Build the `experiment-results` markdown from a parsed results.json."""
    if results is None:
        tail = stdout.strip()[-3000:]
        body = f"```\n{tail}\n```" if tail else "_No structured results were produced._"
        return f"_Executed run produced no `results.json`; raw output below._\n\n{body}\n"

    out: list[str] = ["_Results from real executed code._\n"]
    if results.get("data_note"):  # the agent had to deviate from the planned dataset
        out.append(f"> ⚠️ **Data note:** {results['data_note']}\n")
    for i, exp in enumerate(results.get("experiments", []), 1):
        out.append(f"## Experiment {i}: {exp.get('name', 'untitled')}")
        if exp.get("setup"):
            out.append(f"**Setup:** {exp['setup']}")
        table = _metrics_table(exp.get("metrics", {}) or {})
        if table:
            out.append("**Results:**\n")
            out.extend(table)
        if exp.get("hypothesis"):
            out.append(f"\n**Hypothesis:** {exp['hypothesis']} — **Verdict:** {exp.get('verdict', 'INCONCLUSIVE')}")
        if exp.get("observations"):
            out.append(f"**Observations:** {exp['observations']}")
        out.append(f"**Status:** {exp.get('status', 'MIXED')}\n")

    fig_meta = {f.get("file"): f.get("caption", "") for f in results.get("figures", []) if isinstance(f, dict)}
    if figures:
        out.append("## Figures")
        for fname in figures:
            cap = fig_meta.get(fname, "")
            # bare filename — the frontend resolves it against the run dir via the
            # /api/ideas/{id}/raw file route (figures sit in the run dir root).
            out.append(f"![{cap}]({fname})" + (f"\n*{cap}*" if cap else ""))
        out.append("")
    if results.get("summary"):
        out.append("## Summary")
        out.append(str(results["summary"]))
    return "\n".join(out).rstrip() + "\n"


def _render_failure_md(error: str | None, attempts: int) -> str:
    err = (error or "unknown error").strip()[-2000:]
    return (
        f"_Executed mode could not produce a passing run after {attempts} "
        f"attempt(s). Treat this as an inconclusive/negative experimental "
        f"outcome._\n\n**Last error:**\n```\n{err}\n```\n"
    )


async def run_remote_code(
    settings: LLMSettings,
    idea_ctx: str,
    plan: str,
    out_dir: Path,
    run_config: RunConfig,
    target,
) -> AsyncIterator[dict]:
    """Like :func:`run_local_code` but executes on an SSH remote (GPU box).

    Same generate→run→debug loop; the run step pushes run.py to the remote, runs
    it, and pulls results.json + figures back into ``out_dir``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    label = f"{target.user}@{target.host}"
    messages = [{
        "role": "user",
        "content": f"IDEA spec:\n{idea_ctx}\n\nResearch Plan:\n{plan}\n\nWrite run.py.",
    }]
    last_error: str | None = None

    for attempt in range(1, max(1, run_config.max_attempts) + 1):
        yield {"type": "status", "text": f"\n⚙️ writing experiment code for {label} (attempt {attempt}/{run_config.max_attempts})…\n"}
        raw = ""
        try:
            async for ev in llm.stream_chat_thinking(settings, EXPERIMENT_CODE_SYSTEM, messages, max_tokens=4096):
                if ev["type"] == "thinking":
                    yield {"type": "thinking", "text": ev["text"]}
                else:
                    raw += ev["text"]
                    yield {"type": "delta", "text": ev["text"]}
        except (llm.LLMNotConfigured, llm.LLMError) as exc:
            yield {"type": "result", "result": {
                "markdown": _render_failure_md(str(exc), attempt),
                "provenance": "ssh", "figures": [], "attempts": attempt, "error": str(exc)}}
            return
        code = _extract_code(raw)
        (out_dir / "run.py").write_text(code, encoding="utf-8")

        yield {"type": "status", "text": f"\n▶ executing run.py on {label}…\n"}
        rc, stdout, stderr = await asyncio.to_thread(_exec_remote, target, run_config, out_dir)
        (out_dir / "stdout.log").write_text(
            stdout + ("\n\n--- STDERR ---\n" + stderr if stderr else ""), encoding="utf-8")

        if rc == 0:
            results = _load_results(out_dir)
            figures = sorted(p.name for p in out_dir.glob("*.png"))
            yield {"type": "status", "text": f"✓ run succeeded on {label}\n"}
            yield {"type": "result", "result": {
                "markdown": _render_results_md(results, stdout, figures),
                "provenance": "ssh", "figures": figures, "attempts": attempt, "error": None}}
            return

        last_error = (stderr or stdout)[-_STDERR_TAIL:]
        yield {"type": "status", "text": f"✗ remote run failed (exit {rc}); feeding the error back…\n"}
        messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
        messages.append({"role": "user", "content": EXPERIMENT_CODE_FIX.format(error=last_error)})

    yield {"type": "result", "result": {
        "markdown": _render_failure_md(last_error, run_config.max_attempts),
        "provenance": "ssh", "figures": [], "attempts": run_config.max_attempts, "error": last_error}}


async def generate_figures(
    settings: LLMSettings,
    idea_ctx: str,
    out_dir: Path,
    run_config: RunConfig,
) -> AsyncIterator[dict]:
    """Generate CONCEPTUAL paper figures: LLM writes a matplotlib script, we run
    it (subprocess + timeout) with one fix-retry, and collect the PNGs.

    Yields {"type": "delta"|"status"} progress and a terminal
    {"type": "result", "result": {"figures": [names], "error": str|None}}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    python_exe = run_config.python_path or sys.executable
    messages = [{"role": "user", "content": f"{idea_ctx}\n\nDraw the conceptual figures."}]
    last_error: str | None = None

    for attempt in range(1, 3):
        yield {"type": "status", "text": f"\n🎨 drawing conceptual figures (attempt {attempt}/2)…\n"}
        raw = ""
        try:
            async for ev in llm.stream_chat_thinking(settings, FIGURE_CODE_SYSTEM, messages, max_tokens=3000):
                if ev["type"] == "thinking":
                    yield {"type": "thinking", "text": ev["text"]}
                else:
                    raw += ev["text"]
                    yield {"type": "delta", "text": ev["text"]}
        except (llm.LLMNotConfigured, llm.LLMError) as exc:
            yield {"type": "result", "result": {"figures": [], "error": str(exc)}}
            return
        code = _extract_code(raw)
        (out_dir / "figures.py").write_text(code, encoding="utf-8")
        rc, stdout, stderr = await asyncio.to_thread(
            _exec, python_exe, out_dir, "figures.py", 0  # 0 = no timeout
        )
        if rc == 0:
            figs = sorted(p.name for p in out_dir.glob("*.png"))
            yield {"type": "status", "text": "✓ figures drawn\n"}
            yield {"type": "result", "result": {"figures": figs, "error": None}}
            return
        last_error = (stderr or stdout)[-_STDERR_TAIL:]
        yield {"type": "status", "text": f"✗ figure script failed (exit {rc}); retrying…\n"}
        messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
        messages.append({"role": "user", "content": EXPERIMENT_CODE_FIX.format(error=last_error)})

    yield {"type": "result", "result": {"figures": [], "error": last_error}}


async def run_local_code(
    settings: LLMSettings,
    idea_ctx: str,
    plan: str,
    out_dir: Path,
    run_config: RunConfig,
) -> AsyncIterator[dict]:
    """Generate → execute → debug loop. Yields streaming events (see module docs)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    python_exe = run_config.python_path or sys.executable

    messages = [{
        "role": "user",
        "content": f"IDEA spec:\n{idea_ctx}\n\nResearch Plan:\n{plan}\n\nWrite run.py.",
    }]
    last_error: str | None = None

    for attempt in range(1, max(1, run_config.max_attempts) + 1):
        yield {"type": "status", "text": f"\n⚙️ writing experiment code (attempt {attempt}/{run_config.max_attempts})…\n"}
        raw = ""
        try:
            async for ev in llm.stream_chat_thinking(settings, EXPERIMENT_CODE_SYSTEM, messages, max_tokens=4096):
                if ev["type"] == "thinking":
                    yield {"type": "thinking", "text": ev["text"]}
                else:
                    raw += ev["text"]
                    yield {"type": "delta", "text": ev["text"]}
        except (llm.LLMNotConfigured, llm.LLMError) as exc:
            yield {"type": "result", "result": {
                "markdown": _render_failure_md(str(exc), attempt),
                "provenance": "executed", "figures": [], "attempts": attempt, "error": str(exc),
            }}
            return

        code = _extract_code(raw)
        (out_dir / "run.py").write_text(code, encoding="utf-8")

        yield {"type": "status", "text": "\n▶ executing run.py…\n"}
        rc, stdout, stderr = await asyncio.to_thread(
            _exec, python_exe, out_dir, "run.py", 0  # 0 = no timeout
        )
        (out_dir / "stdout.log").write_text(
            stdout + ("\n\n--- STDERR ---\n" + stderr if stderr else ""), encoding="utf-8"
        )

        if rc == 0:
            results = _load_results(out_dir)
            figures = sorted(p.name for p in out_dir.glob("*.png"))
            yield {"type": "status", "text": "✓ run succeeded\n"}
            yield {"type": "result", "result": {
                "markdown": _render_results_md(results, stdout, figures),
                "provenance": "executed", "figures": figures, "attempts": attempt, "error": None,
            }}
            return

        last_error = (stderr or stdout)[-_STDERR_TAIL:]
        yield {"type": "status", "text": f"✗ run failed (exit {rc}); feeding the error back to fix it…\n"}
        messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
        messages.append({"role": "user", "content": EXPERIMENT_CODE_FIX.format(error=last_error)})

    yield {"type": "result", "result": {
        "markdown": _render_failure_md(last_error, run_config.max_attempts),
        "provenance": "executed", "figures": [], "attempts": run_config.max_attempts, "error": last_error,
    }}
