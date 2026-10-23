"""Real-git regression: fallback clone and SHA checkout agree on CRLF handling.

The uncached fallback path clones a full working tree, then checks out the
locked commit. ``git_network_env`` freezes the host's effective config
(including a Git for Windows ``core.autocrlf=true``) into the clone's child
environment, while the checkout runs under the isolated auth env that hides
that same config. Without a per-command pin the clone writes CRLF, the
checkout compares the tree under LF rules, and ``git checkout <sha>`` refuses
with "Your local changes ... would be overwritten by checkout" whenever the
locked SHA differs from the branch head.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from apm_cli.deps.bare_cache import clone_with_fallback
from apm_cli.utils.git_env import (
    checkout_git_worktree,
    get_git_executable,
    git_subprocess_env,
)

pytestmark = [pytest.mark.component, pytest.mark.windows_compat]

_LF_V1 = b"line one\nline two\n"
_LF_V2 = b"line one\nline two\nline three\n"


def _git(path: Path, *args: str, env: dict[str, str]) -> str:
    result = subprocess.run(
        [get_git_executable(), "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout.strip()


def _neutral_env() -> dict[str, str]:
    env = git_subprocess_env()
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env.pop("GIT_CONFIG_COUNT", None)
    env.pop("GIT_CONFIG_PARAMETERS", None)
    for key in list(env):
        if key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
            env.pop(key, None)
    return env


def _lf_source_with_two_commits(tmp_path: Path) -> tuple[Path, str]:
    """Commit LF bytes twice; return (source, first commit SHA)."""
    env = _neutral_env()
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-b", "main", env=env)
    _git(source, "config", "user.email", "test@example.com", env=env)
    _git(source, "config", "user.name", "Test", env=env)
    payload = source / "payload.txt"
    payload.write_bytes(_LF_V1)
    _git(source, "add", "payload.txt", env=env)
    _git(source, "commit", "-q", "-m", "one", env=env)
    locked_sha = _git(source, "rev-parse", "HEAD", env=env)
    payload.write_bytes(_LF_V2)
    _git(source, "commit", "-q", "-a", "-m", "two", env=env)
    return source, locked_sha


def _apply_host_autocrlf_true(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a system gitconfig with autocrlf=true into the parent process."""
    system_cfg = tmp_path / "system.gitconfig"
    system_cfg.write_text("[core]\n\tautocrlf = true\n", encoding="ascii")
    monkeypatch.delenv("GIT_CONFIG_NOSYSTEM", raising=False)
    monkeypatch.delenv("GIT_CONFIG_PARAMETERS", raising=False)
    monkeypatch.delenv("GIT_CONFIG_COUNT", raising=False)
    for key in list(os.environ):
        if key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(system_cfg))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    observed = subprocess.run(
        [get_git_executable(), "config", "--system", "--get", "core.autocrlf"],
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ),
    )
    assert observed.stdout.strip().lower() == "true"


def _isolated_auth_env() -> dict[str, str]:
    """Mirror the downloader's ``git_env``: system config hidden, empty global."""
    env = git_subprocess_env()
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    return env


def test_fallback_sha_checkout_succeeds_under_host_autocrlf_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, locked_sha = _lf_source_with_two_commits(tmp_path)
    _apply_host_autocrlf_true(tmp_path, monkeypatch)
    isolated_env = _isolated_auth_env()
    target = tmp_path / "dependency"

    def execute_transport_plan(
        repo_url: str,
        target_path: Path,
        *,
        clone_action: Callable[[str, dict[str, str], Path], None],
        **_kwargs: Any,
    ) -> None:
        clone_action(repo_url, isolated_env, target_path)

    repo = clone_with_fallback(execute_transport_plan, str(source), target)
    repo.close()

    # Pre-fix: the clone wrote CRLF under the frozen host config and this
    # checkout refused with "Your local changes ... would be overwritten".
    checkout_git_worktree(target, locked_sha, env=isolated_env)

    assert _git(target, "rev-parse", "HEAD", env=isolated_env) == locked_sha
    assert (target / "payload.txt").read_bytes() == _LF_V1
