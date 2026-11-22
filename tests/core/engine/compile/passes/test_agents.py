"""Test the `agents` module."""

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.engine.compile.passes._agents import AgentsPass


def test_agents_pass_runs_no_rules_of_plain_components():
    """Test that a plain element and an interaction run no rules of their own."""
    assert not AgentsPass._runs_rules(Element('walker', values={'x': 0.0}))
    assert not AgentsPass._runs_rules(Interaction('walk'))
