import asyncio
import subprocess
from pathlib import Path

import pytest

from paperclaw import codex_cli
from paperclaw.config import LLMSettings


def _output_path(args: list[str]) -> Path:
    return Path(args[args.index("--output-last-message") + 1])


def test_codex_prompt_uses_model_and_read_only_temp_dir(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(codex_cli, "codex_executable", lambda: "/usr/bin/codex")

    async def fake_runner(args, input_text, cwd, timeout):
        seen["args"] = args
        seen["input"] = input_text
        seen["cwd"] = cwd
        _output_path(args).write_text("hello from codex", encoding="utf-8")
        return codex_cli.CodexProcessResult(0, stdout='{"type":"assistant_delta","delta":"hello"}\n')

    settings = LLMSettings(provider="codex", model="codex-test-model")
    result = asyncio.run(codex_cli.run_prompt(
        settings, "system", [{"role": "user", "content": "hi"}], runner=fake_runner
    ))

    assert result.text == "hello from codex"
    assert result.model == "codex-test-model"
    assert "--model" in seen["args"] and "codex-test-model" in seen["args"]
    assert seen["args"][seen["args"].index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in seen["args"]
    assert "--skip-git-repo-check" in seen["args"]
    assert "--add-dir" not in seen["args"]
    assert "USER:\nhi" in seen["input"]


def test_codex_workspace_prompt_reports_changed_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(codex_cli, "codex_executable", lambda: "/usr/bin/codex")
    (tmp_path / "IDEA.md").write_text("old", encoding="utf-8")
    seen: dict = {}

    async def fake_runner(args, input_text, cwd, timeout):
        seen["args"] = args
        (cwd / "IDEA.md").write_text("new", encoding="utf-8")
        _output_path(args).write_text("updated", encoding="utf-8")
        return codex_cli.CodexProcessResult(0)

    settings = LLMSettings(provider="codex", model="m")
    result = asyncio.run(codex_cli.run_prompt(
        settings, "system", [{"role": "user", "content": "edit"}],
        workspace_dir=tmp_path,
        allow_writes=True,
        runner=fake_runner,
    ))

    assert result.text == "updated"
    assert result.files_modified == frozenset({"IDEA.md"})
    assert seen["args"][seen["args"].index("--cd") + 1] == str(tmp_path.resolve())
    assert seen["args"][seen["args"].index("--sandbox") + 1] == "workspace-write"


def test_codex_missing_binary_is_not_configured(monkeypatch):
    monkeypatch.setattr(codex_cli, "codex_executable", lambda: None)
    settings = LLMSettings(provider="codex", model="m")

    with pytest.raises(codex_cli.CodexNotConfigured):
        asyncio.run(codex_cli.run_prompt(settings, "system", []))


def test_codex_nonzero_exit_raises_bounded_error(monkeypatch):
    monkeypatch.setattr(codex_cli, "codex_executable", lambda: "/usr/bin/codex")

    async def fake_runner(args, input_text, cwd, timeout):
        return codex_cli.CodexProcessResult(2, stderr="bad things happened")

    settings = LLMSettings(provider="codex", model="m")
    with pytest.raises(codex_cli.CodexError, match="bad things"):
        asyncio.run(codex_cli.run_prompt(settings, "system", [], runner=fake_runner))


def test_codex_readiness_uses_login_status_and_doctor():
    def fake_run(args, timeout):
        if args[1:] == ["login", "status"]:
            return subprocess.CompletedProcess(args, 0, stdout="Logged in using ChatGPT", stderr="")
        if args[1:] == ["doctor"]:
            return subprocess.CompletedProcess(args, 0, stdout="Codex ready", stderr="")
        raise AssertionError(args)

    ready = codex_cli.check_readiness(executable="/usr/bin/codex", runner=fake_run)

    assert ready.installed is True
    assert ready.logged_in is True
    assert ready.healthy is True
    assert "Codex ready" in ready.detail


def test_codex_readiness_reports_logged_out():
    def fake_run(args, timeout):
        return subprocess.CompletedProcess(args, 1, stdout="Not logged in", stderr="")

    ready = codex_cli.check_readiness(executable="/usr/bin/codex", runner=fake_run)

    assert ready.installed is True
    assert ready.logged_in is False
    assert ready.healthy is False
    assert "codex login" in (ready.hint or "")
