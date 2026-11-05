"""Test the `_rule` module of `skcomplex.core.laws`."""

import pytest

from skcomplex.core.laws import RuleLaw
from skcomplex.core.spaces import DiscreteTime
from skcomplex.errors import ValidationError


def test_rule_runs_its_function_on_the_agent_and_the_model():
    """Test that a rule of an element receives the agent and the model, and a rule of an interaction the model."""
    moved, world = [], object()
    RuleLaw(lambda agent, model: moved.append((agent, model))).prepare().run(('walker', world))
    RuleLaw(moved.append).prepare().run(world)
    assert moved == [('walker', world), world]


def test_rule_changes_no_state_and_needs_discrete_time():
    """Test that a rule targets no state, since it changes agents in place, and runs in discrete time."""
    law = RuleLaw(lambda model: None, phase='trade').prepare()
    assert (law.targets_, law.phase) == ((), 'trade')
    assert law._get_requirements().time == (DiscreteTime,)


def test_rule_needs_a_callable_function():
    """Test that a rule rejects a function that cannot be called."""
    with pytest.raises(ValidationError, match='callable'):
        RuleLaw('move').prepare()


@pytest.mark.parametrize('members', ['', 3])
def test_rule_members_name_a_role(members):
    """Test that the members of a rule must name a role."""
    with pytest.raises(ValidationError, match='name a role'):
        RuleLaw(lambda agent, model: None, members=members).prepare()
