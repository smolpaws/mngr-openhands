"""``mngr_openhands`` plugin — registers the ``openhands`` agent type.

OpenHands (https://github.com/OpenHands/OpenHands) is an open-source AI agent
*harness/runtime* — unlike Claude Code / Codex / OpenCode (which mngr already
wraps), OpenHands is its own agent, not a vendor CLI. Its terminal entry point is
the ``openhands`` CLI (a TUI; also ``openhands acp|serve|web``). This plugin makes
``openhands`` a first-class mngr agent type so it can be created, listed, and
orchestrated alongside the others:

    mngr create my-task openhands
    mngr create my-task openhands -- -t "fix the failing test"

STATUS: minimal MVP. This registers the agent type by reusing mngr's built-in
``command`` agent behavior (run the CLI, send keys, unattended). It deliberately
does NOT yet implement the deeper integrations a production plugin like
``mngr_claude`` / ``mngr_opencode`` has — common transcript capture, lifecycle
RUNNING/WAITING markers, permission-prompt detection, per-agent credential
isolation, session adopt/preserve. See README "Roadmap" for the full contract.

The same result is achievable with zero code via a custom agent type in config
(``parent_type = "command"``, ``command = "openhands"``); this package exists so
``openhands`` ships as an installable, shareable type with room to grow into a
full plugin.
"""

from __future__ import annotations

from pydantic import Field

from imbue.mngr import hookimpl
from imbue.mngr.agents.base_agent import SendKeysAgent
from imbue.mngr.config.data_types import AgentTypeConfig
from imbue.mngr.config.data_types import CommandString
from imbue.mngr.interfaces.agent import AgentInterface
from imbue.mngr.interfaces.agent import HasUnattendedModeMixin


class OpenHandsAgentConfig(AgentTypeConfig):
    """Config for the openhands agent type.

    Defaults the base command to ``openhands`` so ``mngr create t openhands``
    works with no extra args (``assemble_command`` reads ``agent_config.command``
    as the base when no override is given).
    """

    command: CommandString = Field(
        default=CommandString("openhands"),
        description="Command to run the OpenHands CLI",
    )


class OpenHandsAgent(SendKeysAgent[OpenHandsAgentConfig], HasUnattendedModeMixin):
    """OpenHands agent type.

    Runs the ``openhands`` CLI in the agent's tmux pane. As a ``SendKeysAgent``,
    ``mngr message`` types into the running TUI; ``mngr connect`` attaches to it.

    Unattended for now (no structured tool-approval bridge yet). When the deeper
    integration lands, this should detect OpenHands' confirmation prompts and
    flip lifecycle to WAITING (cf. ``mngr_opencode`` permission handling).
    """

    def is_unattended_enabled(self) -> bool:
        return True


@hookimpl
def register_agent_type() -> tuple[str, type[AgentInterface] | None, type[AgentTypeConfig]]:
    """Register the ``openhands`` agent type.

    The base command defaults to ``openhands``; ``cli_args`` and any args after
    ``--`` are appended, so e.g. ``mngr create t openhands -- -t "do X"`` runs
    ``openhands -t "do X"``.
    """
    return ("openhands", OpenHandsAgent, OpenHandsAgentConfig)


@hookimpl
def register_agent_aliases() -> dict[str, str]:
    """Register ``oh`` as a short alias for the ``openhands`` agent type."""
    return {"oh": "openhands"}
