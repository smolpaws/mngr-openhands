"""``mngr_openhands`` plugin — registers the ``openhands`` agent type.

OpenHands (https://github.com/OpenHands/OpenHands) is an open-source AI agent
*harness/runtime* — unlike Claude Code / Codex / OpenCode (which mngr already
wraps), OpenHands is its own agent, not a vendor CLI. Its terminal entry point is
the ``openhands`` CLI (a TUI; also ``openhands acp|serve|web``). This plugin makes
``openhands`` a first-class mngr agent type so it can be created, listed, and
orchestrated alongside the others:

    mngr create my-task openhands
    mngr create my-task openhands -- -t "fix the failing test"

How it runs
-----------
OpenHands ships a Textual TUI, so the agent is an :class:`InteractiveTuiAgent`
(like the ``codex`` / ``claude`` plugins): the ``openhands`` CLI runs in the
agent's tmux pane, ``mngr connect`` attaches, and ``mngr message`` pastes into
the TUI and submits with Enter. Readiness is detected from a stable banner in
the TUI (``TUI_READY_INDICATOR``).

Per-agent isolation (the isolation lever)
-----------------------------------------
OpenHands resolves all of its state — settings, credentials, and stored
conversations — from ``OPENHANDS_PERSISTENCE_DIR`` (default ``~/.openhands``),
with ``OPENHANDS_CONVERSATIONS_DIR`` and ``OPENHANDS_WORK_DIR`` as finer
overrides (see ``openhands_cli/locations.py``). This plugin points each agent at
its own persistence dir *under the agent's mngr state dir*, injected only on the
openhands process via :meth:`modify_env_vars`, so parallel agents don't collide
and the user's real ``~/.openhands`` is left untouched. This is the direct
analog of ``mngr_codex``'s per-agent ``CODEX_HOME``.

Shared login: OpenHands stores its LLM settings/credentials in
``agent_settings.json`` inside the persistence dir. To let one login authenticate
every agent, the per-agent ``agent_settings.json`` is symlinked to the user's
shared ``~/.openhands/agent_settings.json`` (opt out with ``share_login=False``).

Unattended: a bare ``openhands`` defaults to *always-ask* confirmation and would
block forever with no TTY driver. When unattended (the default for
orchestration), this appends ``--always-approve`` so the agent runs to
completion. Attended agents keep the interactive confirmation flow.

Still on the roadmap (see README): common-transcript conversion, lifecycle
RUNNING/WAITING detection from confirmation prompts, and session adopt/preserve.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import ClassVar

from loguru import logger
from pydantic import Field

from imbue.mngr import hookimpl
from imbue.mngr.agents.tui_agent import InteractiveTuiAgent
from imbue.mngr.agents.tui_utils import send_enter_best_effort
from imbue.mngr.config.data_types import AgentTypeConfig
from imbue.mngr.config.data_types import CommandString
from imbue.mngr.hosts.common import get_agent_state_dir_path
from imbue.mngr.hosts.common import symlink_on_host
from imbue.mngr.hosts.tmux import TmuxWindowTarget
from imbue.mngr.interfaces.agent import AgentInterface
from imbue.mngr.interfaces.agent import HasUnattendedModeMixin
from imbue.mngr.interfaces.host import OnlineHostInterface

# OpenHands CLI env vars (openhands_cli/locations.py) — the per-agent isolation
# lever. Values default to ~/.openhands; overriding them relocates all state.
PERSISTENCE_DIR_ENV_VAR = "OPENHANDS_PERSISTENCE_DIR"
CONVERSATIONS_DIR_ENV_VAR = "OPENHANDS_CONVERSATIONS_DIR"
WORK_DIR_ENV_VAR = "OPENHANDS_WORK_DIR"

# mngr injects these before ``modify_env_vars`` runs (see host.py).
AGENT_STATE_DIR_ENV_VAR = "MNGR_AGENT_STATE_DIR"
AGENT_WORK_DIR_ENV_VAR = "MNGR_AGENT_WORK_DIR"

# Where, under the agent's mngr state dir, OpenHands' persistence lives.
OPENHANDS_STATE_SUBDIR = "openhands"
CONVERSATIONS_SUBDIR = "conversations"

# The file OpenHands writes its settings + LLM credentials into.
AGENT_SETTINGS_FILENAME = "agent_settings.json"

# Auto-approve flag for unattended runs (a bare TUI defaults to always-ask).
ALWAYS_APPROVE_FLAG = "--always-approve"


class OpenHandsAgentConfig(AgentTypeConfig):
    """Config for the openhands agent type."""

    command: CommandString = Field(
        default=CommandString("openhands"),
        description="Command to run the OpenHands CLI",
    )
    unattended: bool = Field(
        default=True,
        description="Run unattended: append --always-approve so the agent does not "
        "block on OpenHands' interactive confirmation prompts. Set False to keep "
        "the interactive always-ask flow (drive approvals via `mngr connect`).",
    )
    isolate_state: bool = Field(
        default=True,
        description="Give each agent its own OpenHands persistence dir under the "
        "agent's mngr state dir (via OPENHANDS_PERSISTENCE_DIR), so parallel "
        "agents don't share settings/conversations and ~/.openhands is untouched.",
    )
    share_login: bool = Field(
        default=True,
        description="When isolating state, symlink each agent's agent_settings.json "
        "to the user's ~/.openhands/agent_settings.json so one login/LLM config "
        "authenticates every agent. Ignored when isolate_state is False.",
    )


class OpenHandsAgent(
    InteractiveTuiAgent[OpenHandsAgentConfig],
    HasUnattendedModeMixin,
):
    """OpenHands agent type — drives the ``openhands`` TUI in a tmux pane."""

    # Stable text shown in the OpenHands TUI input box once it is ready for
    # input (verified against OpenHands CLI 1.13.1). Used for readiness polling.
    TUI_READY_INDICATOR: ClassVar[str] = "Type your message"

    def get_expected_process_name(self) -> str:
        return "openhands"

    def is_unattended_enabled(self) -> bool:
        return self.agent_config.unattended

    def get_unattended_cli_args(self) -> tuple[str, ...]:
        """CLI args implied by unattended mode (``--always-approve`` when on)."""
        return (ALWAYS_APPROVE_FLAG,) if self.is_unattended_enabled() else ()

    def _send_enter_and_validate(self, tmux_target: TmuxWindowTarget) -> None:
        """Submit the pasted message. OpenHands has no submit hook to wait on
        (unlike codex), so use the best-effort Enter strategy."""
        send_enter_best_effort(self, tmux_target)

    def assemble_command(
        self,
        host: OnlineHostInterface,
        agent_args: tuple[str, ...],
        command_override: CommandString | None,
        initial_message: str | None = None,
    ) -> CommandString:
        """Assemble the ``openhands`` command, injecting unattended flags.

        Unattended flags go right after the base command (before user
        ``cli_args`` / post-``--`` args) so an explicit user override still wins
        by appearing later on the line.
        """
        base = super().assemble_command(host, agent_args, command_override, initial_message)
        extra = self.get_unattended_cli_args()
        if not extra:
            return base
        parts = str(base).split(" ", 1)
        head = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        assembled = " ".join([head, *extra] + ([rest] if rest else []))
        return CommandString(assembled)

    def modify_env_vars(self, host: OnlineHostInterface, env_vars: dict[str, str]) -> None:
        """Point OpenHands' state dirs at this agent's mngr state dir.

        Runs during provisioning after mngr has populated ``MNGR_AGENT_STATE_DIR``
        / ``MNGR_AGENT_WORK_DIR``. No-op when isolation is disabled or the state
        dir is absent (defensive — never invent paths).
        """
        if not self.agent_config.isolate_state:
            return
        agent_state_dir = env_vars.get(AGENT_STATE_DIR_ENV_VAR)
        if not agent_state_dir:
            logger.warning(
                "openhands: {} not set; skipping per-agent isolation", AGENT_STATE_DIR_ENV_VAR
            )
            return

        persistence_dir = os.path.join(agent_state_dir, OPENHANDS_STATE_SUBDIR)
        env_vars[PERSISTENCE_DIR_ENV_VAR] = persistence_dir
        env_vars[CONVERSATIONS_DIR_ENV_VAR] = os.path.join(persistence_dir, CONVERSATIONS_SUBDIR)

        agent_work_dir = env_vars.get(AGENT_WORK_DIR_ENV_VAR)
        if agent_work_dir:
            # Keep file edits in the agent's worktree, not wherever the pane cd'd.
            env_vars[WORK_DIR_ENV_VAR] = agent_work_dir

    def on_before_provisioning(self, host: OnlineHostInterface, options, mngr_ctx) -> None:
        """Create the per-agent OpenHands persistence dir and, if sharing login,
        symlink its ``agent_settings.json`` to the user's shared one."""
        super().on_before_provisioning(host, options, mngr_ctx)
        if not self.agent_config.isolate_state:
            return
        agent_state_dir = self._resolve_agent_state_dir(host)
        if agent_state_dir is None:
            return
        persistence_dir = agent_state_dir / OPENHANDS_STATE_SUBDIR
        host.execute_idempotent_command(f"mkdir -p {shlex.quote(str(persistence_dir))}")

        if self.agent_config.share_login:
            self._link_shared_login(host, persistence_dir)

    # ── helpers ─────────────────────────────────────────────────────────

    def _resolve_agent_state_dir(self, host: OnlineHostInterface) -> Path | None:
        """Best-effort resolve the agent's mngr state dir on ``host``."""
        try:
            return get_agent_state_dir_path(host.host_dir, self.id)
        except Exception as error:  # pragma: no cover - defensive
            logger.warning("openhands: could not resolve agent state dir: {}", error)
            return None

    def _link_shared_login(self, host: OnlineHostInterface, persistence_dir: Path) -> None:
        """Symlink the per-agent ``agent_settings.json`` to the user's shared one
        so a single login/LLM config authenticates every agent."""
        shared = Path(os.path.expanduser("~/.openhands")) / AGENT_SETTINGS_FILENAME
        if not host.path_exists(shared):
            logger.info("openhands: no shared {} found; agent starts with its own config", shared)
            return
        link = persistence_dir / AGENT_SETTINGS_FILENAME
        symlink_on_host(host, source=shared, dest=link, ensure_source_parent=False)


@hookimpl
def register_agent_type() -> tuple[str, type[AgentInterface] | None, type[AgentTypeConfig]]:
    """Register the ``openhands`` agent type.

    The base command defaults to ``openhands``; unattended flags plus ``cli_args``
    and any args after ``--`` are appended, so e.g.
    ``mngr create t openhands -- -t "do X"`` runs
    ``openhands --always-approve -t "do X"``.
    """
    return ("openhands", OpenHandsAgent, OpenHandsAgentConfig)


@hookimpl
def register_agent_aliases() -> dict[str, str]:
    """Register ``oh`` as a short alias for the ``openhands`` agent type."""
    return {"oh": "openhands"}
