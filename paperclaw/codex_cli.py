"""Codex CLI bridge for ChatGPT-authenticated local Codex subscriptions."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable


DEFAULT_TIMEOUT = 600.0


class CodexNotConfigured(Exception):
    pass


class CodexError(Exception):
    pass


@dataclass
class CodexProcessResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass
class CodexResult:
    text: str
    model: str
    files_modified: frozenset[str] = frozenset()
    chunks: tuple[str, ...] = ()


@dataclass
class CodexReadiness:
    installed: bool
    logged_in: bool
    healthy: bool
    detail: str
    hint: str | None = None
    auth_method: str = "unknown"
    auth_configured: bool = False
    auth_candidate: bool = False
    runtime_healthy: bool | None = None
    runtime_detail: str = ""

    @property
    def subscription_auth_configured(self) -> bool:
        """True when ChatGPT-managed auth has been verified."""
        return self.auth_configured or self.logged_in

    @property
    def runtime_ok(self) -> bool:
        return self.healthy if self.runtime_healthy is None else self.runtime_healthy


ProcessRunner = Callable[[list[str], str, Path, float], CodexProcessResult]
SyncRunner = Callable[[list[str], float], subprocess.CompletedProcess]


def codex_executable() -> str | None:
    return shutil.which(os.environ.get("PAPERCLAW_CODEX_BIN", "codex"))


def _snippet(text: str, limit: int = 500) -> str:
    text = (text or "").strip()
    text = _sanitize_diagnostic(text)
    return text[:limit] if len(text) <= limit else text[:limit] + "..."


def _sanitize_diagnostic(text: str) -> str:
    """Redact local paths and secret-shaped values from CLI diagnostics."""
    if not text:
        return ""
    text = re.sub(r"(?i)(sk-[A-Za-z0-9_\-]{8,})", "<redacted>", text)
    text = re.sub(r"(?i)(CODEX_ACCESS_TOKEN|OPENAI_API_KEY|CODEX_API_KEY)=\S+", r"\1=<redacted>", text)
    text = re.sub(r"(?i)(access[_ -]?token|api[_ -]?key|chatgpt[_ -]?token)[\"'=:\s]+[A-Za-z0-9._\-]{12,}", r"\1 <redacted>", text)
    text = re.sub(r"(?i)((?:refresh|id)[_ -]?token|secret)[\"'=:\s]+[A-Za-z0-9._~+/\-=]{8,}", r"\1 <redacted>", text)
    text = re.sub(r"(?i)(bearer)\s+[A-Za-z0-9._~+/\-=]{8,}", r"\1 <redacted>", text)
    text = re.sub(r"(?i)[A-Z]:\\[^\s\"')]+", "<path>", text)
    text = re.sub(r"(?<!\w)/(?:Users|home|private|var|tmp|opt|usr)/[^\s\"')]+", "<path>", text)
    return text


def _detail_bool(details: dict[str, Any], key: str) -> bool:
    value = details.get(key)
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _detail_str(details: dict[str, Any], key: str) -> str:
    return str(details.get(key) or "").strip()


def _env_access_token_present() -> bool:
    return bool(os.environ.get("CODEX_ACCESS_TOKEN"))


def _runtime_detail(checks: dict[str, Any]) -> str:
    problems: list[str] = []
    warnings: list[str] = []
    for check_id, raw in checks.items():
        if check_id == "auth.credentials" or not isinstance(raw, dict):
            continue
        status = str(raw.get("status") or "").lower()
        summary = _sanitize_diagnostic(str(raw.get("summary") or check_id))
        item = f"{check_id}: {summary}"
        if status == "fail":
            problems.append(item)
        elif status == "warning":
            warnings.append(item)
    if problems:
        return "runtime issue: " + "; ".join(problems[:3])
    if warnings:
        return "runtime warning: " + "; ".join(warnings[:3])
    return "runtime checks passed"


def _env_token_candidate(exe_installed: bool, runtime_healthy: bool | None = None) -> CodexReadiness:
    return CodexReadiness(
        installed=exe_installed,
        logged_in=False,
        healthy=False,
        detail="CODEX_ACCESS_TOKEN is set; Codex will validate it when `codex exec` runs",
        hint="persist ChatGPT auth with `codex login --with-access-token`, or run `codex login` for browser/device sign-in",
        auth_method="env_access_token",
        auth_configured=False,
        auth_candidate=True,
        runtime_healthy=runtime_healthy,
        runtime_detail="env token candidate; runtime not verified",
    )


def _readiness_from_login_status(login: subprocess.CompletedProcess, *, exe_installed: bool) -> CodexReadiness:
    login_text = f"{login.stdout}\n{login.stderr}".strip()
    login_low = login_text.lower()
    if login.returncode == 0 and "logged in" in login_low and "chatgpt" in login_low:
        return CodexReadiness(
            installed=exe_installed,
            logged_in=True,
            healthy=True,
            detail=_snippet(login_text) or "Logged in using ChatGPT",
            auth_method="chatgpt",
            auth_configured=True,
            runtime_healthy=True,
            runtime_detail="runtime not checked",
        )
    if login.returncode == 0 and "logged in" in login_low and "api" in login_low:
        return CodexReadiness(
            installed=exe_installed,
            logged_in=False,
            healthy=False,
            detail=_snippet(login_text) or "Codex is logged in with API-key auth",
            hint="run `codex logout`, then `codex login` and choose ChatGPT sign-in",
            auth_method="api_key",
            auth_configured=False,
            runtime_healthy=False,
            runtime_detail="runtime not checked",
        )
    if _env_access_token_present():
        return _env_token_candidate(exe_installed)
    return CodexReadiness(
        installed=exe_installed,
        logged_in=False,
        healthy=False,
        detail=_snippet(login_text) or "Codex is not logged in with ChatGPT",
        hint="run `codex login` and choose ChatGPT sign-in",
        auth_method="none",
        auth_configured=False,
        runtime_healthy=False,
        runtime_detail="runtime not checked",
    )


def _readiness_from_doctor_json(
    payload: dict[str, Any],
    *,
    exe_installed: bool,
    raw_text: str,
) -> CodexReadiness | None:
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        return None
    auth = checks.get("auth.credentials")
    if not isinstance(auth, dict):
        return None
    details = auth.get("details") if isinstance(auth.get("details"), dict) else {}
    summary = _sanitize_diagnostic(str(auth.get("summary") or "auth status unavailable"))
    mode = _detail_str(details, "stored auth mode").lower().replace("-", "_")
    stored_api = _detail_bool(details, "stored API key") or mode in {"api", "api_key", "apikey"}
    stored_chatgpt = _detail_bool(details, "stored ChatGPT tokens") or mode == "chatgpt"
    stored_agent = _detail_bool(details, "stored agent identity") or mode in {
        "access_token",
        "agent_identity",
        "agent",
    }
    overall = str(payload.get("overallStatus") or "").lower()
    runtime_healthy = overall not in {"fail", "error"}
    runtime_detail = _runtime_detail(checks)

    if stored_api:
        return CodexReadiness(
            installed=exe_installed,
            logged_in=False,
            healthy=False,
            detail="Codex is logged in with API-key auth, not ChatGPT subscription auth",
            hint="run `codex logout`, then `codex login` and choose ChatGPT sign-in",
            auth_method="api_key",
            auth_configured=False,
            runtime_healthy=runtime_healthy,
            runtime_detail=runtime_detail,
        )
    if stored_chatgpt or stored_agent:
        method = "access_token" if stored_agent and not stored_chatgpt else "chatgpt"
        return CodexReadiness(
            installed=exe_installed,
            logged_in=True,
            healthy=runtime_healthy,
            detail="Codex auth configured with ChatGPT-managed credentials",
            hint=None if runtime_healthy else "run `codex doctor --json` and fix the reported runtime issue",
            auth_method=method,
            auth_configured=True,
            runtime_healthy=runtime_healthy,
            runtime_detail=runtime_detail,
        )
    if _env_access_token_present():
        return _env_token_candidate(exe_installed, runtime_healthy=runtime_healthy)
    if "not logged" in summary.lower() or str(auth.get("status") or "").lower() == "fail":
        return CodexReadiness(
            installed=exe_installed,
            logged_in=False,
            healthy=False,
            detail=summary,
            hint="run `codex login` and choose ChatGPT sign-in",
            auth_method="none",
            auth_configured=False,
            runtime_healthy=runtime_healthy,
            runtime_detail=runtime_detail,
        )
    return CodexReadiness(
        installed=exe_installed,
        logged_in=False,
        healthy=False,
        detail=_snippet(raw_text) or "Codex auth method is unknown",
        hint="run `codex login status` and `codex doctor --json`",
        auth_method="unknown",
        auth_configured=False,
        runtime_healthy=runtime_healthy,
        runtime_detail=runtime_detail,
    )


def _run_sync(args: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def check_readiness(
    *,
    run_doctor: bool = True,
    executable: str | None = None,
    runner: SyncRunner | None = None,
) -> CodexReadiness:
    """Check Codex CLI availability and ChatGPT login without reading auth files."""
    exe = executable or codex_executable()
    if not exe:
        return CodexReadiness(
            installed=False,
            logged_in=False,
            healthy=False,
            detail="Codex CLI not found on PATH",
            hint="install Codex, then run `codex login` and choose ChatGPT sign-in",
            auth_method="none",
            runtime_healthy=False,
        )

    run = runner or _run_sync
    if run_doctor:
        try:
            doctor = run([exe, "doctor", "--json"], 20.0)
            doctor_text = f"{doctor.stdout}\n{doctor.stderr}".strip()
            try:
                payload = json.loads(doctor.stdout or doctor.stderr or "{}")
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                parsed = _readiness_from_doctor_json(
                    payload,
                    exe_installed=True,
                    raw_text=doctor_text,
                )
                if parsed is not None:
                    return parsed
        except Exception:
            # Older Codex builds may not support `doctor --json`; fall back to
            # the stable login-status probe below.
            pass

    try:
        login = run([exe, "login", "status"], 10.0)
    except Exception as exc:
        return CodexReadiness(
            installed=True,
            logged_in=False,
            healthy=False,
            detail=f"`codex login status` failed: {exc}",
            hint="run `codex login` and choose ChatGPT sign-in",
            auth_method="unknown",
            runtime_healthy=False,
        )

    return _readiness_from_login_status(login, exe_installed=True)


def _snapshot(base_dir: Path) -> dict[str, int]:
    snap: dict[str, int] = {}
    for p in base_dir.rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        try:
            snap[p.relative_to(base_dir).as_posix()] = p.stat().st_mtime_ns
        except OSError:
            pass
    return snap


def _changed(base_dir: Path, before: dict[str, int]) -> frozenset[str]:
    after = _snapshot(base_dir)
    return frozenset(k for k, v in after.items() if before.get(k) != v)


def _prompt(system: str, messages: list[dict]) -> str:
    parts = ["System instructions:", system.strip(), "", "Conversation:"]
    for msg in messages:
        role = str(msg.get("role", "user")).upper()
        parts.extend([f"{role}:", str(msg.get("content", "")).strip(), ""])
    return "\n".join(parts).strip() + "\n"


def _codex_model_arg(model: str | None) -> str:
    from paperclaw.config import DEFAULT_MODEL

    value = (model or "").strip()
    if not value or value == DEFAULT_MODEL:
        return ""
    return value


def _command(exe: str, model: str, cwd: Path, output_file: Path, sandbox: str) -> list[str]:
    args = [
        exe,
        "exec",
        "--cd",
        str(cwd),
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox",
        sandbox,
        "--json",
        "--output-last-message",
        str(output_file),
        "--color",
        "never",
    ]
    model_arg = _codex_model_arg(model)
    if model_arg:
        args.extend(["--model", model_arg])
    args.append("-")
    return args


def _preflight_runtime_auth() -> CodexReadiness:
    ready = check_readiness(run_doctor=False)
    if not ready.installed:
        raise CodexNotConfigured(
            f"{ready.detail}. {ready.hint or 'install Codex, then run `codex login`.'}"
        )
    if ready.subscription_auth_configured or ready.auth_candidate:
        return ready
    hint = f" {ready.hint}" if ready.hint else ""
    raise CodexNotConfigured(f"{ready.detail}.{hint}".strip())


async def _default_runner(
    args: list[str], input_text: str, cwd: Path, timeout: float
) -> CodexProcessResult:
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input_text.encode("utf-8")), timeout=timeout
        )
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise CodexError(f"Codex timed out after {int(timeout)}s") from exc
    return CodexProcessResult(
        returncode=proc.returncode or 0,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
    )


async def _call_runner(
    runner: ProcessRunner,
    args: list[str],
    input_text: str,
    cwd: Path,
    timeout: float,
) -> CodexProcessResult:
    result = runner(args, input_text, cwd, timeout)
    if inspect.isawaitable(result):
        result = await result
    return result


def _jsonl_chunks(stdout: str) -> tuple[str, ...]:
    chunks: list[str] = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        chunk = _event_text(data)
        if chunk:
            chunks.append(chunk)
    return tuple(chunks)


def _event_text(data: dict) -> str:
    event_type = str(data.get("type", "")).lower()
    for key in ("delta", "text"):
        value = data.get(key)
        if isinstance(value, str) and ("delta" in event_type or "assistant" in event_type):
            return value
    message = data.get("message")
    if isinstance(message, str) and "assistant" in event_type:
        return message
    if isinstance(message, dict):
        value = message.get("content") or message.get("text")
        if isinstance(value, str) and "assistant" in event_type:
            return value
    return ""


async def run_prompt(
    settings,
    system: str,
    messages: list[dict],
    *,
    workspace_dir: Path | None = None,
    allow_writes: bool = False,
    runner: ProcessRunner | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> CodexResult:
    exe = codex_executable()
    if not exe:
        raise CodexNotConfigured(
            "Codex CLI not found on PATH. Install Codex, then run `codex login` and choose ChatGPT sign-in."
        )
    _preflight_runtime_auth()

    sandbox = "workspace-write" if allow_writes else "read-only"
    run = runner or _default_runner

    async def _run_in(cwd: Path) -> CodexResult:
        cwd = cwd.resolve()
        before = _snapshot(cwd) if allow_writes else {}
        with tempfile.TemporaryDirectory(prefix="paperclaw-codex-out-") as out_dir:
            output_file = Path(out_dir) / "last-message.txt"
            args = _command(exe, settings.model, cwd, output_file, sandbox)
            result = await _call_runner(run, args, _prompt(system, messages), cwd, timeout)
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            if result.returncode != 0:
                raise CodexError(
                    f"Codex exited with {result.returncode}: {_snippet(stderr or stdout) or 'no output'}"
                )
            chunks = _jsonl_chunks(stdout)
            text = ""
            if output_file.is_file():
                text = output_file.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                text = "".join(chunks).strip() or stdout.strip()
            if not text:
                raise CodexError("Codex returned an empty final response.")
            files = _changed(cwd, before) if allow_writes else frozenset()
            return CodexResult(
                text=text,
                model=_codex_model_arg(settings.model) or "codex-default",
                files_modified=files,
                chunks=chunks,
            )

    if workspace_dir is not None:
        return await _run_in(Path(workspace_dir))

    with tempfile.TemporaryDirectory(prefix="paperclaw-codex-") as tmp:
        return await _run_in(Path(tmp))


async def chat(settings, system: str, messages: list[dict], max_tokens: int = 4096) -> CodexResult:
    return await run_prompt(settings, system, messages)


async def stream_chat(
    settings, system: str, messages: list[dict], max_tokens: int = 4096
) -> AsyncIterator[str]:
    result = await run_prompt(settings, system, messages)
    chunks = result.chunks or (result.text,)
    for chunk in chunks:
        if chunk:
            yield chunk


async def workspace_chat(
    settings,
    base_dir: Path,
    system: str,
    messages: list[dict],
) -> CodexResult:
    return await run_prompt(
        settings,
        system,
        messages,
        workspace_dir=base_dir,
        allow_writes=True,
    )
