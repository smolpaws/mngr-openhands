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
# and have the OpenHands CLI available on PATH:
pip install openhands-ai         # provides the `openhands` command
```

Verify:

```bash
mngr plugin list                 # 'openhands' should appear, ENABLED
```

## Status — MVP

This is a minimal but working plugin. It registers the `openhands` agent type
(and `oh` alias) by reusing mngr's `SendKeysAgent` behavior: the `openhands` CLI
runs in the agent's tmux pane, `mngr connect` attaches, `mngr message` types into
it. It defaults the base command to `openhands` and appends any args after `--`.

It does **not** yet implement the deeper integrations that the production
`mngr_claude` / `mngr_opencode` plugins have. Those are the roadmap.

### Verified (local, mngr 0.2.17, openhands 1.13.1)
- `mngr plugin list` shows the plugin, enabled.
- `mngr create <name> openhands` and the `oh` alias create and start agents.
- An mngr-managed agent successfully executed the OpenHands CLI
  (`OpenHands CLI 1.13.1`).

## Roadmap (toward parity with mngr_claude / mngr_opencode)

To make OpenHands a *first-class citizen* rather than a thin command wrapper:

1. **Common transcript** — convert OpenHands' event stream into mngr's transcript
   format so `mngr transcript` works (OpenHands has structured events / `--json`
   and `openhands view` for trajectories — a good source).
2. **Lifecycle markers** — RUNNING vs WAITING detection, including flipping to
   WAITING on OpenHands' confirmation prompts (cf. mngr_opencode permission
   handling). OpenHands' confirmation mode is the hook point.
3. **Per-agent isolation** — scope OpenHands config/creds/state per agent
   (`~/.openhands/` today) so parallel agents don't collide.
4. **Headless / ACP option** — optionally drive `openhands acp` or `--headless`
   for non-TTY orchestration instead of the TUI.
5. **Session adopt/preserve** — map OpenHands conversation ids to mngr's
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
