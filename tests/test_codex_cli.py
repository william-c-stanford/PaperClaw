import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from paperclaw import codex_cli
from paperclaw.config import LLMSettings


def _output_path(args: list[str]) -> Path:
    return Path(args[args.index("--output-last-message") + 1])


def _ready(**kw) -> codex_cli.CodexReadiness:
    data = {
        "installed": True,
        "logged_in": True,
        "healthy": True,
        "detail": "ready",
        "auth_method": "chatgpt",
        "auth_configured": True,
        "runtime_healthy": True,
    }
    data.update(kw)
    return codex_cli.CodexReadiness(**data)


def test_codex_prompt_uses_model_and_read_only_temp_dir(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(codex_cli, "codex_executable", lambda: "/usr/bin/codex")
    monkeypatch.setattr(codex_cli, "check_readiness", lambda run_doctor=True: _ready())

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
    monkeypatch.setattr(codex_cli, "check_readiness", lambda run_doctor=True: _ready())
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
    monkeypatch.setattr(codex_cli, "check_readiness", lambda run_doctor=True: _ready())

    async def fake_runner(args, input_text, cwd, timeout):
        return codex_cli.CodexProcessResult(2, stderr="bad things happened")

    settings = LLMSettings(provider="codex", model="m")
    with pytest.raises(codex_cli.CodexError, match="bad things"):
        asyncio.run(codex_cli.run_prompt(settings, "system", [], runner=fake_runner))


def test_codex_command_omits_inherited_anthropic_default(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(codex_cli, "codex_executable", lambda: "/usr/bin/codex")
    monkeypatch.setattr(codex_cli, "check_readiness", lambda run_doctor=True: _ready())

    async def fake_runner(args, input_text, cwd, timeout):
        seen["args"] = args
        _output_path(args).write_text("default model", encoding="utf-8")
        return codex_cli.CodexProcessResult(0)

    settings = LLMSettings(provider="codex")
    result = asyncio.run(codex_cli.run_prompt(
        settings, "system", [{"role": "user", "content": "hi"}], runner=fake_runner
    ))

    assert result.model == "codex-default"
    assert "--model" not in seen["args"]


def test_codex_runtime_preflight_rejects_api_key_auth(monkeypatch):
    monkeypatch.setattr(codex_cli, "codex_executable", lambda: "/usr/bin/codex")
    monkeypatch.setattr(
        codex_cli,
        "check_readiness",
        lambda run_doctor=True: _ready(
            logged_in=False,
            healthy=False,
            detail="Codex is logged in with API-key auth",
            hint="run `codex logout`",
            auth_method="api_key",
            auth_configured=False,
            runtime_healthy=False,
        ),
    )
    called = False

    async def fake_runner(args, input_text, cwd, timeout):
        nonlocal called
        called = True
        return codex_cli.CodexProcessResult(0)

    settings = LLMSettings(provider="codex", model="m")
    with pytest.raises(codex_cli.CodexNotConfigured, match="API-key auth"):
        asyncio.run(codex_cli.run_prompt(settings, "system", [], runner=fake_runner))
    assert called is False


def test_codex_runtime_preflight_allows_env_access_token_candidate(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(codex_cli, "codex_executable", lambda: "/usr/bin/codex")
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "secret-token-value")
    monkeypatch.setattr(
        codex_cli,
        "check_readiness",
        lambda run_doctor=True: _ready(
            logged_in=False,
            healthy=False,
            detail="CODEX_ACCESS_TOKEN is set",
            auth_method="env_access_token",
            auth_configured=False,
            auth_candidate=True,
            runtime_healthy=None,
        ),
    )

    async def fake_runner(args, input_text, cwd, timeout):
        seen["input"] = input_text
        _output_path(args).write_text("token ok", encoding="utf-8")
        return codex_cli.CodexProcessResult(0)

    settings = LLMSettings(provider="codex", model="codex-test-model")
    result = asyncio.run(codex_cli.run_prompt(settings, "system", [], runner=fake_runner))

    assert result.text == "token ok"
    assert "secret-token-value" not in seen["input"]


def test_codex_readiness_parses_doctor_json_chatgpt_with_runtime_failure():
    payload = {
        "overallStatus": "fail",
        "checks": {
            "auth.credentials": {
                "status": "ok",
                "summary": "auth is configured",
                "details": {
                    "stored auth mode": "chatgpt",
                    "stored API key": "false",
                    "stored ChatGPT tokens": "true",
                    "stored agent identity": "false",
                    "auth file": "/Users/me/.codex/auth.json",
                },
            },
            "network.provider_reachability": {
                "status": "fail",
                "summary": "ChatGPT base URL connect failed",
            },
        },
    }

    def fake_run(args, timeout):
        if args[1:] == ["doctor", "--json"]:
            return subprocess.CompletedProcess(args, 1, stdout=json.dumps(payload), stderr="")
        raise AssertionError(args)

    ready = codex_cli.check_readiness(executable="/usr/bin/codex", runner=fake_run)

    assert ready.installed is True
    assert ready.logged_in is True
    assert ready.subscription_auth_configured is True
    assert ready.healthy is False
    assert ready.runtime_healthy is False
    assert ready.auth_method == "chatgpt"
    assert "auth.json" not in ready.detail
    assert "network.provider_reachability" in ready.runtime_detail


def test_codex_readiness_reports_api_key_auth_from_doctor_json():
    payload = {
        "overallStatus": "ok",
        "checks": {
            "auth.credentials": {
                "status": "ok",
                "summary": "auth is configured",
                "details": {
                    "stored auth mode": "api",
                    "stored API key": "true",
                    "stored ChatGPT tokens": "false",
                },
            }
        },
    }

    def fake_run(args, timeout):
        if args[1:] == ["doctor", "--json"]:
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload), stderr="")
        raise AssertionError(args)

    ready = codex_cli.check_readiness(executable="/usr/bin/codex", runner=fake_run)

    assert ready.logged_in is False
    assert ready.subscription_auth_configured is False
    assert ready.auth_method == "api_key"
    assert "API-key auth" in ready.detail
    assert "codex logout" in (ready.hint or "")


def test_codex_readiness_reports_access_token_auth_from_doctor_json():
    payload = {
        "overallStatus": "ok",
        "checks": {
            "auth.credentials": {
                "status": "ok",
                "summary": "auth is configured",
                "details": {
                    "stored auth mode": "agent",
                    "stored API key": "false",
                    "stored ChatGPT tokens": "false",
                    "stored agent identity": "true",
                },
            }
        },
    }

    def fake_run(args, timeout):
        if args[1:] == ["doctor", "--json"]:
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload), stderr="")
        raise AssertionError(args)

    ready = codex_cli.check_readiness(executable="/usr/bin/codex", runner=fake_run)

    assert ready.logged_in is True
    assert ready.auth_method == "access_token"
    assert ready.subscription_auth_configured is True


def test_codex_readiness_sanitizes_secret_like_diagnostics():
    payload = {
        "overallStatus": "fail",
        "checks": {
            "auth.credentials": {
                "status": "fail",
                "summary": "refresh_token=refresh-secret-value123 bearer abcdefghijklmnop /Users/me/.codex/auth.json",
                "details": {},
            },
        },
    }

    def fake_run(args, timeout):
        if args[1:] == ["doctor", "--json"]:
            return subprocess.CompletedProcess(args, 1, stdout=json.dumps(payload), stderr="")
        raise AssertionError(args)

    ready = codex_cli.check_readiness(executable="/usr/bin/codex", runner=fake_run)

    assert "refresh-secret" not in ready.detail
    assert "abcdefghijklmnop" not in ready.detail
    assert "/Users/me" not in ready.detail


def test_codex_readiness_falls_back_to_login_status_when_doctor_json_is_bad():
    def fake_run(args, timeout):
        if args[1:] == ["doctor", "--json"]:
            return subprocess.CompletedProcess(args, 1, stdout="not-json", stderr="")
        if args[1:] == ["login", "status"]:
            return subprocess.CompletedProcess(args, 0, stdout="Logged in using ChatGPT", stderr="")
        raise AssertionError(args)

    ready = codex_cli.check_readiness(executable="/usr/bin/codex", runner=fake_run)

    assert ready.logged_in is True
    assert ready.auth_method == "chatgpt"


def test_codex_readiness_reports_api_key_auth_from_login_status():
    def fake_run(args, timeout):
        if args[1:] == ["login", "status"]:
            return subprocess.CompletedProcess(args, 0, stdout="Logged in using API key", stderr="")
        raise AssertionError(args)

    ready = codex_cli.check_readiness(
        executable="/usr/bin/codex",
        runner=fake_run,
        run_doctor=False,
    )

    assert ready.logged_in is False
    assert ready.auth_method == "api_key"
    assert "codex logout" in (ready.hint or "")


def test_codex_readiness_reports_logged_out():
    def fake_run(args, timeout):
        return subprocess.CompletedProcess(args, 1, stdout="Not logged in", stderr="")

    ready = codex_cli.check_readiness(executable="/usr/bin/codex", runner=fake_run)

    assert ready.installed is True
    assert ready.logged_in is False
    assert ready.healthy is False
    assert "codex login" in (ready.hint or "")
