"""Hook-shape tests — no live mngr runtime needed."""

from imbue.mngr_openhands import plugin


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
