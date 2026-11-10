"""Test the `agents` module."""

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.engine.compile.passes._agents import AgentsPass


def test_agents_pass_runs_rules():
    """Test `AgentsPass._runs_rules` on a small case."""

    class WalkerElement(Element):
        def _get_laws(self, namespace):
            return [RuleLaw(lambda agent, system: None)]

    assert (AgentsPass._runs_rules(WalkerElement('walker')), AgentsPass._runs_rules(Element('plain'))) == (True, False)
