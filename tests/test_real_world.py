"""Real-world integration test: drive a live OpenHands agent through mngr.

This is deliberately end-to-end — it runs the actual ``mngr`` CLI, which creates
a git worktree and launches the real ``openhands`` CLI in a tmux pane, and it
talks to a real LLM. It verifies the two headline behaviors of this plugin:

1. **Per-agent isolation** — the agent's OpenHands state (settings +
   conversation) lands under the agent's own mngr state dir
   (``OPENHANDS_PERSISTENCE_DIR``), not the user's ``~/.openhands``.
2. **Shared login + unattended run** — with ``share_login`` on, the per-agent
   ``agent_settings.json`` is symlinked to the user's shared one, so the agent
   authenticates with the user's existing LLM config and runs to completion
   (the plugin injects ``--always-approve`` so it never blocks on confirmation).
   The agent edits a file in its worktree, proving the whole path works.

It is **opt-in**: it does nothing on a normal ``pytest`` run so the unit suite
stays fast, free, and runnable everywhere. It only runs when you explicitly ask
for it *and* a usable shared login exists:

    MNGR_OPENHANDS_REAL_WORLD=1 pytest tests/test_real_world.py -s

Requirements when opted in (otherwise it skips, never fails):

- ``mngr``, ``openhands``, ``tmux`` and ``git`` on ``PATH``.
- A usable shared login at ``~/.openhands/agent_settings.json`` (or point
  ``OPENHANDS_SHARED_SETTINGS`` at one) containing a real LLM ``api_key``. This
  is what the plugin's ``share_login`` mode symlinks into each agent, so the
  test exercises exactly that path — no ad-hoc credentials in the environment.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest

REQUIRED_TOOLS = ("mngr", "openhands", "tmux", "git")
RUN_TIMEOUT_SECONDS = 300
OPT_IN_ENV_VAR = "MNGR_OPENHANDS_REAL_WORLD"
SHARED_SETTINGS_ENV_VAR = "OPENHANDS_SHARED_SETTINGS"
DEFAULT_SHARED_SETTINGS = Path.home() / ".openhands" / "agent_settings.json"


def _missing_tools() -> list[str]:
    return [tool for tool in REQUIRED_TOOLS if shutil.which(tool) is None]


def _shared_settings_path() -> Path:
    override = os.environ.get(SHARED_SETTINGS_ENV_VAR)
    return Path(override) if override else DEFAULT_SHARED_SETTINGS


def _shared_login_usable() -> bool:
    """True only if the shared settings file exists and carries a real API key.

    The plugin's ``share_login`` symlinks this exact file into each agent, so a
    usable one is what makes an unattended live run possible. We never read or
    log the key itself — only confirm one is present.
    """
    path = _shared_settings_path()
    if not path.is_file():
        return False
    try:
        settings = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    api_key = (settings.get("llm") or {}).get("api_key")
    # A masked ("***") or empty key means the file can't actually authenticate.
    return isinstance(api_key, str) and bool(api_key.strip()) and set(api_key) != {"*"}


pytestmark = [
    pytest.mark.skipif(
        os.environ.get(OPT_IN_ENV_VAR) != "1",
        reason=f"opt-in only: set {OPT_IN_ENV_VAR}=1 to run the live end-to-end test",
    ),
    pytest.mark.skipif(bool(_missing_tools()), reason=f"missing tools: {_missing_tools()}"),
    pytest.mark.skipif(
        not _shared_login_usable(),
        reason=f"no usable shared login at {_shared_settings_path()} (need a real llm.api_key)",
    ),
]


def _run(cmd: list[str], cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init", "-q"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    _run(["git", "config", "user.name", "Test"], cwd=repo)
    (repo / "README.md").write_text("# test repo\n")
    _run(["git", "add", "-A"], cwd=repo)
    _run(["git", "commit", "-q", "-m", "init"], cwd=repo)
    return repo


def test_mngr_runs_isolated_shared_login_openhands_agent(git_repo: Path):
    """Create an openhands agent via mngr that uses the shared login, runs
    unattended, and edits a file in its own isolated worktree/state dir."""
    agent_name = f"ohtest{uuid.uuid4().hex[:8]}"
    sentinel = f"share-ok-{uuid.uuid4().hex[:6]}"
    task = (
        f"Create a file named result.txt in the current directory containing "
        f"exactly the text '{sentinel}', then finish."
    )

    # No --env / --override-with-envs here on purpose: the agent must pick up the
    # user's shared login that the plugin symlinks in (share_login defaults on).
    # The plugin injects --always-approve so the unattended run won't block.
    create = _run(
        [
            "mngr", "create", agent_name, "openhands",
            "--no-connect", "--yes",
            "--", "-t", task,
        ],
        cwd=git_repo,
        timeout=120,
    )
    assert create.returncode == 0, (
        f"mngr create failed:\nSTDOUT:{create.stdout}\nSTDERR:{create.stderr}"
    )

    try:
        deadline = time.time() + RUN_TIMEOUT_SECONDS
        worktree = None
        result_file = None
        while time.time() < deadline:
            if worktree is None:
                worktree = _resolve_worktree(git_repo, agent_name)
            if worktree is not None:
                candidate = worktree / "result.txt"
                if candidate.exists() and sentinel in candidate.read_text():
                    result_file = candidate
                    break
            time.sleep(5)

        assert result_file is not None, (
            f"agent did not produce result.txt with sentinel in "
            f"{RUN_TIMEOUT_SECONDS}s (worktree={worktree})"
        )

        # Isolation: the conversation persisted under the agent's own mngr state
        # dir, NOT the user's ~/.openhands.
        state_dir = _resolve_agent_state_dir(agent_name)
        assert state_dir is not None, "could not resolve agent mngr state dir"
        conversations = state_dir / "openhands" / "conversations"
        assert conversations.is_dir(), (
            f"expected isolated conversations dir at {conversations}"
        )
        assert any(conversations.iterdir()), (
            "no conversation persisted under the agent state dir"
        )

        # Shared login: the per-agent settings is the symlink the plugin created.
        settings_link = state_dir / "openhands" / "agent_settings.json"
        assert settings_link.is_symlink(), (
            f"expected shared-login symlink at {settings_link}"
        )
    finally:
        _run(["mngr", "destroy", agent_name, "--yes"], cwd=git_repo, timeout=60)


def _resolve_worktree(repo: Path, agent_name: str) -> Path | None:
    """Find the git worktree mngr created for the agent (path contains its name)."""
    result = _run(["git", "worktree", "list", "--porcelain"], cwd=repo)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            path = Path(line[len("worktree "):].strip())
            if agent_name in str(path):
                return path
    return None


def _resolve_agent_state_dir(agent_name: str) -> Path | None:
    """Find the agent's mngr state dir via its isolated persistence subdir."""
    mngr_home = Path(os.environ.get("MNGR_HOME", Path.home() / ".mngr"))
    if not mngr_home.exists():
        return None
    for conversations in mngr_home.rglob("openhands/conversations"):
        state_dir = conversations.parent.parent
        if agent_name in str(state_dir):
            return state_dir
    # Fall back to the first match if the name isn't encoded in the path.
    for conversations in mngr_home.rglob("openhands/conversations"):
        return conversations.parent.parent
    return None
