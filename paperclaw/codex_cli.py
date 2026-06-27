"""Codex CLI bridge for ChatGPT-authenticated local Codex subscriptions."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable


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


ProcessRunner = Callable[[list[str], str, Path, float], CodexProcessResult]
SyncRunner = Callable[[list[str], float], subprocess.CompletedProcess]


def codex_executable() -> str | None:
    return shutil.which(os.environ.get("PAPERCLAW_CODEX_BIN", "codex"))


def _snippet(text: str, limit: int = 500) -> str:
    text = (text or "").strip()
    return text[:limit] if len(text) <= limit else text[:limit] + "..."


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
        )

    run = runner or _run_sync
    try:
        login = run([exe, "login", "status"], 10.0)
    except Exception as exc:
        return CodexReadiness(
            installed=True,
            logged_in=False,
            healthy=False,
            detail=f"`codex login status` failed: {exc}",
            hint="run `codex login` and choose ChatGPT sign-in",
        )

    login_text = f"{login.stdout}\n{login.stderr}".strip()
    login_low = login_text.lower()
    logged_in = login.returncode == 0 and "logged in" in login_low and "chatgpt" in login_low
    if not logged_in:
        return CodexReadiness(
            installed=True,
            logged_in=False,
            healthy=False,
            detail=_snippet(login_text) or "Codex is not logged in with ChatGPT",
            hint="run `codex login` and choose ChatGPT sign-in",
        )

    if not run_doctor:
        return CodexReadiness(
            installed=True,
            logged_in=True,
            healthy=True,
            detail=_snippet(login_text) or "Logged in using ChatGPT",
        )

    try:
        doctor = run([exe, "doctor"], 20.0)
    except Exception as exc:
        return CodexReadiness(
            installed=True,
            logged_in=True,
            healthy=False,
            detail=f"`codex doctor` failed: {exc}",
            hint="run `codex doctor` for details",
        )

    doctor_text = f"{doctor.stdout}\n{doctor.stderr}".strip()
    if doctor.returncode != 0:
        return CodexReadiness(
            installed=True,
            logged_in=True,
            healthy=False,
            detail=_snippet(doctor_text) or "`codex doctor` reported a failure",
            hint="run `codex doctor` and fix the reported Codex issue",
        )

    return CodexReadiness(
        installed=True,
        logged_in=True,
        healthy=True,
        detail=_snippet(doctor_text) or "Codex CLI ready",
    )


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
    if model:
        args.extend(["--model", model])
    args.append("-")
    return args


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
                model=settings.model,
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
