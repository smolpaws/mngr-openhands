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
        state_dir = _resolve_agent_state_dir(git_repo, agent_name)
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
        _run(["mngr", "destroy", agent_name, "-f"], cwd=git_repo, timeout=60)


def test_preserve_then_adopt_carries_conversation_forward(git_repo: Path):
    """End-to-end: an agent's conversation is preserved on destroy, then adopted
    into a fresh agent, which relaunches with ``--resume`` against it."""
    agent_a = f"ohsrc{uuid.uuid4().hex[:8]}"
    sentinel = f"adopt-{uuid.uuid4().hex[:6]}"
    task = (
        f"Create a file named result.txt in the current directory containing "
        f"exactly the text '{sentinel}', then finish."
    )

    create = _run(
        ["mngr", "create", agent_a, "openhands", "--no-connect", "--yes", "--", "-t", task],
        cwd=git_repo,
        timeout=120,
    )
    assert create.returncode == 0, (
        f"mngr create failed:\nSTDOUT:{create.stdout}\nSTDERR:{create.stderr}"
    )

    conversation_id = None
    try:
        # Wait for a conversation to be persisted under agent A's isolated store.
        state_dir = None
        deadline = time.time() + RUN_TIMEOUT_SECONDS
        while time.time() < deadline:
            if state_dir is None:
                state_dir = _resolve_agent_state_dir(git_repo, agent_a)
            if state_dir is not None:
                conversations = state_dir / "openhands" / "conversations"
                ids = [p.name for p in conversations.iterdir()] if conversations.is_dir() else []
                if ids:
                    conversation_id = ids[0]
                    break
            time.sleep(5)
        assert conversation_id is not None, "agent A never persisted a conversation"
    finally:
        # Destroy A with preservation on (the default) so its conversation survives.
        _run(["mngr", "destroy", agent_a, "-f"], cwd=git_repo, timeout=60)

    # The conversation must now live under mngr's preserved/ area.
    preserved = _find_preserved_conversation(conversation_id)
    assert preserved is not None, (
        f"conversation {conversation_id} was not preserved after destroy"
    )

    # Adopt it into a fresh agent B.
    agent_b = f"ohdst{uuid.uuid4().hex[:8]}"
    create_b = _run(
        [
            "mngr", "create", agent_b, "openhands",
            "--no-connect", "--yes", "--adopt", conversation_id,
        ],
        cwd=git_repo,
        timeout=120,
    )
    assert create_b.returncode == 0, (
        f"mngr create --adopt failed:\nSTDOUT:{create_b.stdout}\nSTDERR:{create_b.stderr}"
    )
    try:
        state_b = _resolve_agent_state_dir(git_repo, agent_b)
        assert state_b is not None, "could not resolve agent B state dir"
        # The adopted conversation was copied into B's isolated store...
        adopted = state_b / "openhands" / "conversations" / conversation_id
        assert adopted.is_dir(), f"adopted conversation not copied into {adopted}"
        # ...and B recorded a resume pointer to it.
        pointer = state_b / "openhands_resume_conversation"
        assert pointer.is_file() and pointer.read_text().strip() == conversation_id, (
            f"expected resume pointer for {conversation_id} at {pointer}"
        )
    finally:
        _run(["mngr", "destroy", agent_b, "-f"], cwd=git_repo, timeout=60)


def _find_preserved_conversation(conversation_id: str) -> Path | None:
    """Locate a preserved conversation dir by id under mngr's preserved/ area."""
    mngr_home = Path(os.environ.get("MNGR_HOME", Path.home() / ".mngr"))
    preserved_root = mngr_home / "preserved"
    if not preserved_root.is_dir():
        return None
    for candidate in preserved_root.rglob(f"conversations/{conversation_id}"):
        if candidate.is_dir():
            return candidate
    return None


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


def _resolve_agent_state_dir(repo: Path, agent_name: str) -> Path | None:
    """Resolve *this* agent's mngr state dir (``<mngr_home>/agents/<id>``).

    mngr keys the state dir by the agent's generated id, not its name, so we ask
    mngr for the id of the agent we created (matched by name) and build the path
    from that — never by scanning for the first ``openhands`` dir, which could
    match an unrelated agent and turn the isolation check into a false positive.
    """
    # Parse stdout regardless of exit code: mngr may return non-zero if an
    # unrelated provider (e.g. Docker) is unreachable while still listing the
    # local agents we care about on stdout.
    result = _run(["mngr", "list", "--fields", "id,name"], cwd=repo)
    agent_id = None
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == agent_name and parts[0].startswith("agent-"):
            agent_id = parts[0]
            break
    if agent_id is None:
        return None
    mngr_home = Path(os.environ.get("MNGR_HOME", Path.home() / ".mngr"))
    state_dir = mngr_home / "agents" / agent_id
    return state_dir if state_dir.exists() else None
