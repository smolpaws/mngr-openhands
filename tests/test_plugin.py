"""Unit tests for the mngr_openhands plugin — no live mngr runtime needed.

These exercise the pure/logic surface: hook shapes, config defaults, the
per-agent isolation env mapping, and the unattended command assembly. The
live end-to-end behavior is covered separately in ``test_real_world.py``.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from imbue.mngr.agents.tui_agent import InteractiveTuiAgent

from imbue.mngr_openhands import plugin


# ── Hook shape ───────────────────────────────────────────────────────────


def test_register_agent_type_shape():
    name, agent_cls, config_cls = plugin.register_agent_type()
    assert name == "openhands"
    assert agent_cls is plugin.OpenHandsAgent
    assert config_cls is plugin.OpenHandsAgentConfig


def test_default_command_is_openhands():
    assert str(plugin.OpenHandsAgentConfig().command) == "openhands"


def test_alias_maps_oh_to_openhands():
    aliases = plugin.register_agent_aliases()
    assert aliases == {"oh": "openhands"}


# ── Base class / TUI wiring ──────────────────────────────────────────────


def test_agent_is_a_tui_agent():
    # OpenHands ships a TUI (``openhands``), so it should be driven as an
    # interactive TUI agent (proper readiness + paste-based send), not a bare
    # send-keys command wrapper.
    assert issubclass(plugin.OpenHandsAgent, InteractiveTuiAgent)


def test_tui_ready_indicator_is_set():
    # A concrete InteractiveTuiAgent must declare a stable readiness banner.
    indicator = plugin.OpenHandsAgent.TUI_READY_INDICATOR
    assert isinstance(indicator, str) and indicator.strip()


def test_expected_process_name_is_openhands():
    assert _agent().get_expected_process_name() == "openhands"


# ── Config flags ─────────────────────────────────────────────────────────


def test_isolation_defaults_on():
    cfg = plugin.OpenHandsAgentConfig()
    assert cfg.isolate_state is True
    assert cfg.share_login is True


# ── Per-agent isolation (the headline feature) ───────────────────────────


def _agent(**cfg_kwargs):
    """Build an OpenHandsAgent for pure-logic tests.

    Uses ``model_construct`` to bypass the runtime fields (host, mngr_ctx, ...)
    that a live agent needs — the methods under test only read ``agent_config``.
    """
    cfg = plugin.OpenHandsAgentConfig(**cfg_kwargs)
    return plugin.OpenHandsAgent.model_construct(agent_type="openhands", agent_config=cfg)


def test_isolation_env_points_openhands_dirs_under_agent_state_dir():
    agent = _agent()
    env = {"MNGR_AGENT_STATE_DIR": "/host/state/agent-1", "MNGR_AGENT_WORK_DIR": "/host/work/agent-1"}
    agent.modify_env_vars(host=None, env_vars=env)  # host unused by our logic

    persist = env["OPENHANDS_PERSISTENCE_DIR"]
    assert persist.startswith("/host/state/agent-1")
    # conversations live under the per-agent persistence dir
    assert env["OPENHANDS_CONVERSATIONS_DIR"].startswith("/host/state/agent-1")
    # work dir tracks the agent's work dir so file edits land in the worktree
    assert env["OPENHANDS_WORK_DIR"] == "/host/work/agent-1"


def test_isolation_disabled_leaves_env_untouched():
    agent = _agent(isolate_state=False)
    env = {"MNGR_AGENT_STATE_DIR": "/host/state/agent-1"}
    agent.modify_env_vars(host=None, env_vars=env)
    assert "OPENHANDS_PERSISTENCE_DIR" not in env


def test_isolation_noop_without_agent_state_dir():
    # Defensive: if mngr didn't provide the state dir, don't invent paths.
    agent = _agent()
    env: dict[str, str] = {}
    agent.modify_env_vars(host=None, env_vars=env)
    assert "OPENHANDS_PERSISTENCE_DIR" not in env


# ── Unattended command assembly ──────────────────────────────────────────


def test_unattended_appends_always_approve():
    # A bare ``openhands`` defaults to always-ask and would hang unattended;
    # unattended mode must auto-approve so orchestration doesn't stall.
    agent = _agent()
    assert agent.is_unattended_enabled() is True
    extra = agent.get_unattended_cli_args()
    assert "--always-approve" in extra


def test_attended_does_not_auto_approve():
    agent = _agent(unattended=False)
    assert agent.is_unattended_enabled() is False
    assert "--always-approve" not in agent.get_unattended_cli_args()


# These assert the unattended-flag placement in isolation from the resume-prelude
# wrapping (covered by the resume-prelude tests), so isolate_state is off here —
# which makes _wrap_with_resume_prelude a no-op and keeps the command exact.


def test_assemble_command_unattended_inserts_flag_after_base():
    agent = _agent(isolate_state=False)
    cmd = str(agent.assemble_command(host=None, agent_args=("-t", "do X"), command_override=None))
    # flag right after the base command, before user args (which still win if repeated later)
    assert cmd == "openhands --always-approve -t 'do X'"


def test_assemble_command_attended_is_bare():
    agent = _agent(unattended=False, isolate_state=False)
    cmd = str(agent.assemble_command(host=None, agent_args=(), command_override=None))
    assert cmd == "openhands"


def test_assemble_command_handles_multi_word_base():
    # A multi-word base command (e.g. a launcher) must keep the flag attached to
    # the openhands invocation, not wedged between the launcher's words.
    agent = _agent(command="poetry run openhands", isolate_state=False)
    cmd = str(agent.assemble_command(host=None, agent_args=("-t", "do X"), command_override=None))
    assert cmd == "poetry run openhands --always-approve -t 'do X'"


def test_assemble_command_override_wins_and_keeps_flag():
    # An explicit command_override replaces the base; the unattended flag still
    # lands right after it.
    agent = _agent(isolate_state=False)
    cmd = str(
        agent.assemble_command(
            host=None,
            agent_args=(),
            command_override=plugin.CommandString("uv run openhands"),
        )
    )
    assert cmd == "uv run openhands --always-approve"


# ── Shared-login home resolution (remote-safe) ───────────────────────────


class _FakeHost:
    """Minimal host stand-in for the shared-login boundary.

    A test double for the external *host* (which may be remote), not for the
    code under test: it records the symlink command and reports a home dir that
    differs from the local machine's, so we prove ``_host_home`` asks the host
    rather than resolving ``~`` locally.
    """

    def __init__(self, home: str, existing_paths: set[str]):
        self._home = home
        self._existing = existing_paths
        self.commands: list[str] = []
        self.host_dir = plugin.Path(f"{home}/.mngr")

    def execute_idempotent_command(self, command, **kwargs):
        self.commands.append(command)
        return SimpleNamespace(stdout=self._home, stderr="", success=True)

    def path_exists(self, path) -> bool:
        return str(path) in self._existing


def test_host_home_resolves_from_host_not_local():
    host = _FakeHost(home="/home/remoteuser", existing_paths=set())
    home = plugin.OpenHandsAgent._host_home(host)
    assert home == plugin.Path("/home/remoteuser")


def test_shared_login_links_against_host_home():
    home = "/home/remoteuser"
    shared = f"{home}/.openhands/agent_settings.json"
    host = _FakeHost(home=home, existing_paths={shared})
    agent = _agent()
    agent._link_shared_login(host, plugin.Path(f"{home}/.mngr/agents/a1/openhands"))
    # The symlink command must target the host-resolved shared settings path.
    link_cmds = [c for c in host.commands if "ln -sfn" in c]
    assert link_cmds, "expected a symlink command"
    assert shared in link_cmds[0]


def test_shared_login_skips_when_no_shared_settings():
    host = _FakeHost(home="/home/remoteuser", existing_paths=set())
    agent = _agent()
    agent._link_shared_login(host, plugin.Path("/home/remoteuser/.mngr/agents/a1/openhands"))
    assert not [c for c in host.commands if "ln -sfn" in c]


# ── Session preserve / adopt ─────────────────────────────────────────────


def test_supports_session_preservation_and_adoption():
    from imbue.mngr.interfaces.agent import HasSessionAdoptionMixin
    from imbue.mngr.interfaces.agent import HasSessionPreservationMixin

    assert issubclass(plugin.OpenHandsAgent, HasSessionPreservationMixin)
    assert issubclass(plugin.OpenHandsAgent, HasSessionAdoptionMixin)


def test_preserve_on_destroy_defaults_on():
    assert plugin.OpenHandsAgentConfig().preserve_on_destroy is True


def test_preserved_items_is_the_conversations_dir():
    items = plugin._openhands_preserved_items()
    assert len(items) == 1
    assert items[0].rel_path == plugin.CONVERSATIONS_RELPATH
    # a whole tree, not a single file
    assert items[0].kind.name == "DIRECTORY"


class _AdoptHost:
    """Test double for the host during adoption: records copies + written files."""

    def __init__(self, latest_conversation: str | None = None):
        self.copies: list[tuple[Path, Path]] = []
        self.written: dict[str, str] = {}
        self._latest = latest_conversation

    def copy_directory(self, source_host, source_path, target_path, *a, **k) -> None:
        self.copies.append((Path(source_path), Path(target_path)))

    def write_text_file(self, path, content, *a, **k) -> None:
        self.written[str(path)] = content

    def path_exists(self, path) -> bool:
        # Used by transfer_cloned_agent_session_store for the source store.
        return self._latest is not None

    def execute_idempotent_command(self, command, **kwargs):
        # Backs _find_latest_conversation_id's `ls -1t`.
        return SimpleNamespace(stdout=self._latest or "", stderr="", success=True)


def _adopt_agent(monkeypatch, state_dir: Path, **cfg):
    cfg_obj = plugin.OpenHandsAgentConfig(**cfg)
    agent = plugin.OpenHandsAgent.model_construct(
        agent_type="openhands", agent_config=cfg_obj, id="agent-test"
    )
    # _host_agent_dir resolves the state dir on the host; pin it to our tmp dir
    # (both the host-resolve and the local fallback) so the test drives our path.
    monkeypatch.setattr(type(agent), "_resolve_agent_state_dir", lambda self, host: state_dir)
    monkeypatch.setattr(type(agent), "_get_agent_dir", lambda self: state_dir)
    return agent


def test_resume_prelude_wraps_command_when_isolated(monkeypatch, tmp_path):
    agent = _adopt_agent(monkeypatch, tmp_path / "st")
    wrapped = str(
        agent._wrap_with_resume_prelude(_AdoptHost(), plugin.CommandString("openhands --always-approve"))
    )
    assert "--resume" in wrapped
    assert "openhands --always-approve" in wrapped
    assert str(tmp_path / "st" / plugin.RESUME_POINTER_FILENAME) in wrapped


def test_resume_prelude_is_noop_when_not_isolated(monkeypatch, tmp_path):
    agent = _adopt_agent(monkeypatch, tmp_path / "st", isolate_state=False)
    cmd = plugin.CommandString("openhands")
    assert str(agent._wrap_with_resume_prelude(_AdoptHost(), cmd)) == "openhands"


def test_resolve_adopt_conversation_absolute_path(tmp_path):
    conv = tmp_path / "conversations" / "abc123"
    conv.mkdir(parents=True)
    cid, source = plugin._resolve_adopt_conversation(str(conv), mngr_ctx=None)
    assert cid == "abc123"
    assert source == conv.resolve()


def test_resolve_adopt_conversation_absolute_path_missing(tmp_path):
    from imbue.mngr.errors import UserInputError

    with pytest.raises(UserInputError):
        plugin._resolve_adopt_conversation(str(tmp_path / "nope"), mngr_ctx=None)


def test_resolve_adopt_conversation_bare_id_searches_stores(monkeypatch, tmp_path):
    store = tmp_path / "agentX" / "openhands" / "conversations"
    (store / "conv-42").mkdir(parents=True)
    monkeypatch.setattr(plugin, "_mngr_conversation_dirs", lambda ctx: [store])
    cid, source = plugin._resolve_adopt_conversation("conv-42", mngr_ctx=None)
    assert cid == "conv-42"
    assert source == store / "conv-42"


def test_resolve_adopt_conversation_bare_id_not_found(monkeypatch, tmp_path):
    from imbue.mngr.errors import UserInputError

    monkeypatch.setattr(plugin, "_mngr_conversation_dirs", lambda ctx: [tmp_path / "empty"])
    with pytest.raises(UserInputError):
        plugin._resolve_adopt_conversation("missing", mngr_ctx=None)


def test_adopt_explicit_copies_conversation_and_writes_resume_pointer(monkeypatch, tmp_path):
    state_dir = tmp_path / "state"
    source_conv = tmp_path / "src" / "conversations" / "conv-7"
    source_conv.mkdir(parents=True)
    monkeypatch.setattr(plugin, "_mngr_conversation_dirs", lambda ctx: [source_conv.parent])

    agent = _adopt_agent(monkeypatch, state_dir)
    host = _AdoptHost()
    options = SimpleNamespace(adopt_session=("conv-7",), source_agent_state_location=None)
    agent.adopt_session(host, options, mngr_ctx=None)

    # copied into the agent's isolated conversations dir under the same id
    dest = state_dir / plugin.CONVERSATIONS_RELPATH / "conv-7"
    assert (source_conv, dest) in host.copies
    # resume pointer records the adopted id
    pointer = str(state_dir / plugin.RESUME_POINTER_FILENAME)
    assert host.written.get(pointer) == "conv-7"


def test_adopt_noop_when_not_isolated(monkeypatch, tmp_path):
    agent = _adopt_agent(monkeypatch, tmp_path / "state", isolate_state=False)
    host = _AdoptHost()
    options = SimpleNamespace(adopt_session=("conv-7",), source_agent_state_location=None)
    agent.adopt_session(host, options, mngr_ctx=None)
    assert not host.copies
    assert not host.written


def test_adopt_nothing_when_no_options(monkeypatch, tmp_path):
    agent = _adopt_agent(monkeypatch, tmp_path / "state")
    host = _AdoptHost()
    options = SimpleNamespace(adopt_session=(), source_agent_state_location=None)
    agent.adopt_session(host, options, mngr_ctx=None)
    assert not host.copies
    assert not host.written
