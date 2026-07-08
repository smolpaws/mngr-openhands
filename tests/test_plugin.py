"""Unit tests for the mngr_openhands plugin — no live mngr runtime needed.

These exercise the pure/logic surface: hook shapes, config defaults, the
per-agent isolation env mapping, and the unattended command assembly. The
live end-to-end behavior is covered separately in ``test_real_world.py``.
"""

from types import SimpleNamespace

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
    agent.modify_env_vars(host=None, env_vars=dict(env))
    after = {"MNGR_AGENT_STATE_DIR": "/host/state/agent-1"}
    agent.modify_env_vars(host=None, env_vars=after)
    assert "OPENHANDS_PERSISTENCE_DIR" not in after


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


def test_assemble_command_unattended_inserts_flag_after_base():
    agent = _agent()
    cmd = str(agent.assemble_command(host=None, agent_args=("-t", "do X"), command_override=None))
    # flag right after the base command, before user args (which still win if repeated later)
    assert cmd == "openhands --always-approve -t 'do X'"


def test_assemble_command_attended_is_bare():
    agent = _agent(unattended=False)
    cmd = str(agent.assemble_command(host=None, agent_args=(), command_override=None))
    assert cmd == "openhands"


def test_assemble_command_handles_multi_word_base():
    # A multi-word base command (e.g. a launcher) must keep the flag attached to
    # the openhands invocation, not wedged between the launcher's words.
    agent = _agent(command="poetry run openhands")
    cmd = str(agent.assemble_command(host=None, agent_args=("-t", "do X"), command_override=None))
    assert cmd == "poetry run openhands --always-approve -t 'do X'"


def test_assemble_command_override_wins_and_keeps_flag():
    # An explicit command_override replaces the base; the unattended flag still
    # lands right after it.
    agent = _agent()
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
