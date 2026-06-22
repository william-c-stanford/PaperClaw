"""Shared test fixtures."""

import pytest

from paperclaw.server import store as _store
from paperclaw.server.models import RunConfig


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_run_config: don't force the simulated experiment default")


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Point PAPERCLAW_HOME at a tmp dir so no test writes into the real ./saves default,
    and run in a clean cwd so a project-root `settings.yaml` / `.env` doesn't leak into
    `load_settings` (which reads ./settings.yaml + ./.env from the working directory)."""
    monkeypatch.setenv("PAPERCLAW_HOME", str(tmp_path / "_paperclaw_home"))
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _simulated_experiments(request, monkeypatch):
    """Default experiments to SIMULATED in tests — hermetic, no real subprocess.

    Production auto-defaults to `cli` (claude) / `executed` when nothing is saved
    (`store.default_run_config`), which would spawn real agents/jobs during tests.
    Tests that want a specific mode still `store.save_run_config(...)` explicitly;
    mark a test `real_run_config` to exercise the actual auto-default logic.
    """
    if request.node.get_closest_marker("real_run_config"):
        return
    monkeypatch.setattr(_store, "default_run_config",
                        lambda: RunConfig(experimentMode="simulated"))
