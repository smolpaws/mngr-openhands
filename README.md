# mngr-openhands

An [mngr](https://github.com/imbue-ai/mngr) plugin that registers **OpenHands** as
a first-class agent type, so you can create, list, and orchestrate OpenHands
agents alongside Claude Code, Codex, and OpenCode.

```bash
mngr create my-task openhands
mngr create my-task openhands -- -t "fix the failing test in test_foo.py"
mngr create my-task oh           # 'oh' is a registered alias
mngr list                        # OpenHands agents show up like any other
```

## Why

mngr is a provider-agnostic orchestration layer over coding-agent CLIs. Its
built-in plugins wrap **Claude Code**, **Codex**, and **OpenCode** — all vendor
CLIs. [OpenHands](https://github.com/OpenHands/OpenHands) is different: it's an
open-source agent *harness/runtime* in its own right, with a terminal CLI
(`openhands`, plus `openhands acp|serve|web`). This plugin slots OpenHands into
mngr's plugin system so the same `mngr create / list / connect / message /
snapshot / clone` workflow drives OpenHands too.

## Install

```bash
pip install imbue-mngr
pip install mngr-openhands        # (or: pip install -e . from a checkout)
# and have the OpenHands CLI on PATH (the V1 agent-sdk CLI):
uv tool install openhands        # recommended; provides the `openhands` command
# or: pip install openhands
```

Verify:

```bash
mngr plugin list                 # 'openhands' should appear, ENABLED
```

## How it works

OpenHands ships a Textual TUI, so the plugin drives it as an
`InteractiveTuiAgent` (like the built-in `codex` / `claude` plugins): the
`openhands` CLI runs in the agent's tmux pane, `mngr connect` attaches, and
`mngr message` pastes into the TUI and submits with Enter. Readiness is detected
from a stable banner in the TUI.

Three behaviors make OpenHands usable under orchestration, all on by default and
each toggleable via config:

- **Per-agent isolation** (`isolate_state`) — OpenHands resolves all of its
  state (settings, credentials, stored conversations) from
  `OPENHANDS_PERSISTENCE_DIR` (default `~/.openhands`). The plugin points each
  agent at its own persistence dir under the agent's mngr state dir — injected
  only on the openhands process — so parallel agents don't collide and your real
  `~/.openhands` is left untouched. This mirrors `mngr_codex`'s per-agent
  `CODEX_HOME`.
- **Shared login** (`share_login`) — so you don't have to re-authenticate every
  agent, the per-agent `agent_settings.json` is symlinked to your shared
  `~/.openhands/agent_settings.json`. One login authenticates every agent. Opt
  out to give each agent its own config.
- **Unattended** (`unattended`) — a bare `openhands` defaults to *always-ask*
  confirmation and would block forever with no TTY driver, so unattended mode
  appends `--always-approve` and the agent runs to completion. Turn it off to
  keep the interactive approval flow (drive approvals via `mngr connect`).

### Verified (local, mngr 0.2.17, openhands 1.13.1)
- `mngr plugin list` shows the plugin, enabled.
- `mngr create <name> openhands` and the `oh` alias create and start agents.
- End-to-end (`tests/test_real_world.py`): `mngr create` launches a real
  OpenHands agent that uses the shared login, runs unattended, edits a file in
  its own git worktree, and persists its conversation under the isolated
  per-agent state dir (not `~/.openhands`).

## Roadmap (toward parity with mngr_claude / mngr_opencode)

Remaining to make OpenHands a full first-class citizen:

1. **Common transcript** — convert OpenHands' event stream into mngr's transcript
   format so `mngr transcript` works (OpenHands has structured events / `--json`
   and `openhands view` for trajectories — a good source).
2. **Lifecycle markers** — RUNNING vs WAITING detection, including flipping to
   WAITING on OpenHands' confirmation prompts (cf. mngr_opencode permission
   handling). OpenHands' confirmation mode is the hook point.
3. **Headless / ACP option** — optionally drive `openhands acp` or `--headless`
   for non-TTY orchestration instead of the TUI.
4. **Session adopt/preserve** — map OpenHands conversation ids to mngr's
   adopt/preserve on create/destroy.

## Zero-code alternative

You don't strictly need this package — mngr can run OpenHands today via a custom
agent type in config:

```toml
# mngr config edit  (or --scope project)
[agent_types.openhands]
parent_type = "command"
command = "openhands"
```

This package exists so `openhands` ships as an installable, shareable type with
room to grow into the full integration above.

## License

MIT.
