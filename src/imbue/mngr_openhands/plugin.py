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

Session preserve/adopt (conversation portability)
-------------------------------------------------
OpenHands stores each conversation as a directory under
``OPENHANDS_CONVERSATIONS_DIR`` (``<persistence_dir>/conversations/<id>``), and
the ``openhands`` CLI resumes one with ``--resume <id>`` / ``--last``. Because a
resumed conversation takes its working dir from ``OPENHANDS_WORK_DIR`` (which
this plugin already injects) rather than from the stored state, no cwd rebinding
is needed — unlike codex.

- **Preserve** (``preserve_on_destroy``): on destroy, the agent's whole
  ``conversations`` tree is copied to mngr's ``preserved/`` area before the
  state dir is removed, so the conversation survives the agent.
- **Adopt** (``--adopt <id>`` / ``--from <agent>``): a freshly created agent can
  resume an existing conversation. The named conversation dir is copied into the
  new agent's isolated store and ``--resume <id>`` is appended to its launch
  command. ``--adopt`` names a conversation explicitly (an unknown id is an
  error); ``--from`` carries the source agent's latest conversation forward as a
  bonus (a source with none just starts fresh).

Still on the roadmap (see README): common-transcript conversion and lifecycle
RUNNING/WAITING detection from confirmation prompts.
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
from imbue.mngr.api.preservation import PreservedItem
from imbue.mngr.api.preservation import adopt_sessions
from imbue.mngr.api.preservation import dedupe_by_resolved_path
from imbue.mngr.api.preservation import flag_gated_items
from imbue.mngr.api.preservation import iter_agent_session_paths
from imbue.mngr.api.preservation import preserve_agent_state
from imbue.mngr.api.preservation import preserve_host_agents_on_destroy
from imbue.mngr.api.preservation import require_unique_match
from imbue.mngr.api.preservation import run_adopt_session_preflight
from imbue.mngr.api.preservation import transfer_cloned_agent_session_store
from imbue.mngr.config.data_types import AgentTypeConfig
from imbue.mngr.config.data_types import CommandString
from imbue.mngr.config.data_types import MngrContext
from imbue.mngr.hosts.common import get_agent_state_dir_path
from imbue.mngr.hosts.common import symlink_on_host
from imbue.mngr.hosts.tmux import TmuxWindowTarget
from imbue.mngr.interfaces.agent import AgentInterface
from imbue.mngr.interfaces.agent import HasSessionAdoptionMixin
from imbue.mngr.interfaces.agent import HasSessionPreservationMixin
from imbue.mngr.interfaces.agent import HasUnattendedModeMixin
from imbue.mngr.interfaces.data_types import FileType
from imbue.mngr.interfaces.host import CreateAgentOptions
from imbue.mngr.interfaces.host import HostInterface
from imbue.mngr.interfaces.host import HostLocation
from imbue.mngr.interfaces.host import OnlineHostInterface
from imbue.mngr.plugins.hookspecs import OnBeforeCreateArgs
from imbue.mngr.primitives import AgentTypeName
from imbue.mngr.primitives import DiscoveredAgent
from imbue.mngr.errors import UserInputError

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

# The user's shared OpenHands home dir (where a normal `openhands` login lives).
SHARED_OPENHANDS_DIRNAME = ".openhands"

# The file OpenHands writes its settings + LLM credentials into.
AGENT_SETTINGS_FILENAME = "agent_settings.json"

# Auto-approve flag for unattended runs (a bare TUI defaults to always-ask).
ALWAYS_APPROVE_FLAG = "--always-approve"

# Resume an existing conversation (openhands_cli/entrypoint.py: --resume <id>).
RESUME_FLAG = "--resume"

# Conversations live at ``<state_subdir>/conversations`` under the agent state
# dir; this relpath addresses that store for preserve + adopt (mirrors codex's
# ``sessions`` relpath).
CONVERSATIONS_RELPATH = os.path.join(OPENHANDS_STATE_SUBDIR, CONVERSATIONS_SUBDIR)

# Stable event-source name for this agent type (preserve/transcript convention).
OPENHANDS_EVENT_SOURCE = "openhands"

# Per-agent file recording the conversation id to resume, written by
# ``adopt_session`` and read by ``assemble_command`` (analog of codex's
# ``codex_root_session``). Lives directly under the agent state dir.
RESUME_POINTER_FILENAME = "openhands_resume_conversation"


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
    preserve_on_destroy: bool = Field(
        default=True,
        description="When destroying this agent, first copy its OpenHands conversations "
        "to <local_host_dir>/preserved/ so they survive. Set to False to discard them.",
    )


class OpenHandsAgent(
    InteractiveTuiAgent[OpenHandsAgentConfig],
    HasUnattendedModeMixin,
    HasSessionPreservationMixin,
    HasSessionAdoptionMixin,
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

        The unattended flags are appended to the *base command* (the override if
        given, else ``agent_config.command``) before delegating to the parent's
        assembly. That places them right after the base — before user
        ``cli_args`` / post-``--`` args, so an explicit user override still wins
        by appearing later — while letting the parent handle multi-word base
        commands (e.g. ``poetry run openhands``) and arg quoting correctly.
        """
        extra = self.get_unattended_cli_args()
        if extra:
            base_cmd = command_override or self.agent_config.command
            if base_cmd:
                command_override = CommandString(f"{base_cmd} {' '.join(extra)}")
        base = super().assemble_command(host, agent_args, command_override, initial_message)
        return self._wrap_with_resume_prelude(base)

    def _wrap_with_resume_prelude(self, base: CommandString) -> CommandString:
        """Append ``--resume <id>`` when an adopted conversation is pending.

        ``adopt_session`` runs in ``on_after_provisioning`` — *after*
        ``assemble_command`` — so the resume id is not known when the command is
        built. Like codex, the id is therefore read at launch from a per-agent
        pointer file (shell-evaluated, since the stored command is replayed on
        every ``mngr start``). ``set --`` appends ``--resume <id>`` without
        unquoted word-splitting, and an empty/absent pointer leaves ``"$@"``
        empty (a plain, unresumed launch). No-op unless state is isolated (the
        pointer lives under the isolated state dir).
        """
        if not self.agent_config.isolate_state:
            return base
        quoted_pointer = shlex.quote(str(self._get_agent_dir() / RESUME_POINTER_FILENAME))
        resume_prelude = (
            f'__oh_cid="$(cat {quoted_pointer} 2>/dev/null || true)"; set --; '
            f'if [ -n "$__oh_cid" ]; then set -- {RESUME_FLAG} "$__oh_cid"; fi'
        )
        return CommandString(f'{{ {resume_prelude}; {base} "$@"; }}')

    # ── session preserve / adopt ─────────────────────────────────────────

    def on_destroy(self, host: OnlineHostInterface) -> None:
        """Preserve the agent's conversations before its state dir is deleted."""
        if self.agent_config.preserve_on_destroy:
            self.preserve_session_state(host)

    def preserve_session_state(self, host: OnlineHostInterface) -> None:
        preserve_agent_state(_openhands_preserved_items(), self, host)

    def on_after_provisioning(
        self,
        host: OnlineHostInterface,
        options: CreateAgentOptions,
        mngr_ctx: MngrContext,
    ) -> None:
        """Adopt an existing conversation so the new agent resumes it."""
        self.adopt_session(host, options, mngr_ctx)

    def adopt_session(
        self,
        host: OnlineHostInterface,
        options: CreateAgentOptions,
        mngr_ctx: MngrContext,
    ) -> None:
        """Adopt existing OpenHands conversation(s) into this newly provisioned agent.

        Two sources, combined via the shared ``adopt_sessions`` orchestrator:

        - ``--adopt`` (``options.adopt_session``): each value is a conversation id
          (or an absolute path to a conversation dir), resolved against the user's
          native store and every live/preserved mngr agent, then copied into this
          agent's isolated ``conversations`` dir. An unknown id is a hard error.
        - ``--from <agent>`` (``options.source_agent_state_location``): a clone that
          copies the source workspace but not its state dir; the source agent's
          whole conversations tree is transferred in and its latest conversation
          resumed. A source with no conversation is a warning, not an error.

        The conversation actually resumed — via the pointer file that
        ``assemble_command``'s prelude reads — is the ``--from`` clone's when given,
        else the last ``--adopt`` value. With neither set, nothing is adopted
        (fresh start). No cwd rebinding is needed: OpenHands takes the resumed
        conversation's working dir from ``OPENHANDS_WORK_DIR`` (injected per-agent).
        """
        if not self.agent_config.isolate_state:
            if options.adopt_session or options.source_agent_state_location is not None:
                logger.warning(
                    "openhands: session adoption requires isolate_state=True; ignoring --adopt/--from"
                )
            return
        adopt_sessions(
            options.adopt_session,
            options.source_agent_state_location,
            copy_explicit=lambda arg: self._copy_explicit_conversation(host, arg, mngr_ctx),
            copy_clone=lambda location: self._copy_cloned_conversation(host, location),
            resume=lambda conversation_id: self._write_resume_pointer(host, conversation_id),
        )

    def _agent_conversations_dir(self) -> Path:
        """This agent's isolated conversations dir (``<state>/openhands/conversations``)."""
        return self._get_agent_dir() / CONVERSATIONS_RELPATH

    def _copy_explicit_conversation(
        self, host: OnlineHostInterface, adopt_arg: str, mngr_ctx: MngrContext
    ) -> str:
        """Resolve one ``--adopt`` value, copy its conversation dir in, return its id."""
        conversation_id, source_conversation_dir = _resolve_adopt_conversation(adopt_arg, mngr_ctx)
        dest = self._agent_conversations_dir() / conversation_id
        host.copy_directory(host, source_conversation_dir, dest)
        logger.info("Adopted OpenHands conversation {} into agent {}", conversation_id, self.id)
        return conversation_id

    def _copy_cloned_conversation(
        self, host: OnlineHostInterface, source_location: HostLocation
    ) -> str | None:
        """Transfer a cloned source agent's conversations in; resume its latest one."""
        copied = transfer_cloned_agent_session_store(
            host, self._get_agent_dir(), source_location, Path(CONVERSATIONS_RELPATH)
        )
        if not copied:
            logger.info("openhands: cloned source has no conversations; starting fresh")
            return None
        return self._find_latest_conversation_id(host, self._agent_conversations_dir())

    def _write_resume_pointer(self, host: OnlineHostInterface, conversation_id: str) -> None:
        """Record the conversation id ``assemble_command``'s prelude resumes on launch."""
        pointer = self._get_agent_dir() / RESUME_POINTER_FILENAME
        host.write_text_file(pointer, conversation_id)

    @staticmethod
    def _find_latest_conversation_id(host: OnlineHostInterface, conversations_dir: Path) -> str | None:
        """Return the id (dir name) of the most-recently-modified conversation, or None."""
        result = host.execute_idempotent_command(
            f"ls -1t {shlex.quote(str(conversations_dir))} 2>/dev/null || true"
        )
        for line in result.stdout.splitlines():
            name = line.strip()
            if name:
                return name
        return None

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
        so a single login/LLM config authenticates every agent.

        The shared path must resolve on *the host the agent runs on*, which may
        be remote — so the home dir is queried from the host itself, never from
        the local machine's ``~``.
        """
        shared = self._host_home(host) / SHARED_OPENHANDS_DIRNAME / AGENT_SETTINGS_FILENAME
        if not host.path_exists(shared):
            logger.info("openhands: no shared {} found; agent starts with its own config", shared)
            return
        link = persistence_dir / AGENT_SETTINGS_FILENAME
        symlink_on_host(host, source=shared, dest=link, ensure_source_parent=False)

    @staticmethod
    def _host_home(host: OnlineHostInterface) -> Path:
        """Resolve the home directory *on ``host``* (which may be remote).

        Asks the host for ``$HOME`` directly so the shared-login path is correct
        on remote hosts (SSH/Docker/Modal), where the local machine's home dir
        (e.g. ``/Users/x``) does not exist. Falls back to the mngr state dir's
        parent, then the local home — best-effort, never raising.
        """
        try:
            result = host.execute_idempotent_command('printf %s "$HOME"')
            home = result.stdout.strip()
            if home:
                return Path(home)
        except Exception as error:  # pragma: no cover - defensive
            logger.warning("openhands: could not resolve $HOME on host: {}", error)
        try:
            return Path(host.host_dir).parent
        except Exception:  # pragma: no cover - defensive
            return Path(os.path.expanduser("~"))


# ── session preserve / adopt: module-level helpers ───────────────────────

# The conversations store, addressed relative to the agent state dir (the same
# relpath used for preserve and for scanning adopt sources).
_CONVERSATIONS_RELPATH: Path = Path(CONVERSATIONS_RELPATH)


def _openhands_preserved_items() -> list[PreservedItem]:
    """Files to preserve from an OpenHands agent's state dir: its conversations.

    (No transcript items yet — this plugin does not emit a common/raw transcript;
    when it does, add ``build_transcript_preserved_items("openhands")`` here.)
    """
    return [PreservedItem(rel_path=CONVERSATIONS_RELPATH, kind=FileType.DIRECTORY)]


def _openhands_items_for_discovered_agent(ref: DiscoveredAgent):
    """Items to preserve for a discovered (offline) openhands agent, or None to skip."""
    return flag_gated_items(ref, "preserve_on_destroy", _openhands_preserved_items())


def _mngr_conversation_dirs(mngr_ctx: MngrContext) -> list[Path]:
    """Per-agent OpenHands ``conversations`` dirs across live + preserved local agents."""
    local_host_dir = Path(mngr_ctx.config.default_host_dir).expanduser()
    return iter_agent_session_paths(local_host_dir, _CONVERSATIONS_RELPATH)


def _resolve_adopt_conversation(adopt_arg: str, mngr_ctx: MngrContext) -> tuple[str, Path]:
    """Resolve an ``--adopt`` value to a ``(conversation_id, source_conversation_dir)`` pair.

    Accepts either:

    - An absolute path to a conversation directory (its basename is the id).
    - A bare conversation id, searched across every live and preserved local mngr
      agent's ``conversations`` store. An id present in more than one store is
      rejected as ambiguous (pass the absolute path to disambiguate).
    """
    candidate = Path(adopt_arg)
    if candidate.is_absolute():
        source = candidate.resolve()
        if not source.is_dir():
            raise UserInputError(f"Conversation directory not found: {source}")
        return source.name, source

    matches: list[Path] = []
    for conversations_dir in dedupe_by_resolved_path(_mngr_conversation_dirs(mngr_ctx)):
        candidate_dir = conversations_dir / adopt_arg
        if candidate_dir.is_dir():
            matches.append(candidate_dir)

    matched = require_unique_match(
        matches,
        not_found_message=(
            f"OpenHands conversation {adopt_arg} not found. Check the id, or pass an absolute "
            "path to the conversation directory. (Searched every live and preserved mngr "
            "openhands agent.)"
        ),
        ambiguous_message=(
            f"OpenHands conversation {adopt_arg} found in multiple stores; pass the absolute "
            "path to the conversation directory to specify which one:"
        ),
    )
    return adopt_arg, matched


@hookimpl
def on_before_create(args: OnBeforeCreateArgs, mngr_ctx: MngrContext) -> OnBeforeCreateArgs | None:
    """Fail fast on bad ``--adopt`` conversation ids before any host/worktree exists."""
    run_adopt_session_preflight(
        args.agent_options.agent_type,
        args.agent_options.adopt_session,
        mngr_ctx,
        OpenHandsAgent,
        resolve_one=lambda arg: _resolve_adopt_conversation(arg, mngr_ctx),
    )
    return None


@hookimpl
def on_before_host_destroy(host: HostInterface, mngr_ctx: MngrContext) -> None:
    """Preserve openhands conversations off a host's volume before it is destroyed.

    Mirrors ``OpenHandsAgent.on_destroy`` for the offline path, where a host is
    destroyed without per-agent ``on_destroy`` calls but agent state still lives
    on the host's persisted volume.
    """
    preserve_host_agents_on_destroy(
        host, mngr_ctx, AgentTypeName("openhands"), _openhands_items_for_discovered_agent
    )


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
