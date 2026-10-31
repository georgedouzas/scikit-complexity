"""Test the `types` module of `skcomplex.core`."""

import typing
from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from skcomplex.core.types import AgentViewProtocol, MatchingRule, Process, Target
from skcomplex.specs import GreedyMatchingSpec, RandomSearchSpec


def test_targets_are_symbolic_expressions_or_vectors():
    """Test that a law targets a symbolic expression or a vector of them."""
    assert typing.get_args(Target.__value__) == (Expression, FreeModuleElement)


def test_processes_may_be_nothing():
    """Test that a process may be nothing."""
    assert type(None) in typing.get_args(Process.__value__)


def test_agent_views_have_an_id_a_location_and_a_life():
    """Test that an object with the id, location, and moves of an agent is an agent view, and an id alone is not."""
    names = ('neighbor_cells', 'agents_here', 'distance_to', 'move_to', 'remove', 'create')
    view = SimpleNamespace(
        id=1,
        group='walkers',
        location=(0, 0),
        location_key=(0, 0),
        alive=True,
        **dict.fromkeys(names, print),
    )
    assert isinstance(view, AgentViewProtocol)
    assert not isinstance(SimpleNamespace(id=1), AgentViewProtocol)


def test_matching_rules_are_the_matching_specs():
    """Test that a matching rule is one of the specs of the matching algorithms."""
    rules = typing.get_args(MatchingRule.__value__)
    assert RandomSearchSpec in rules
    assert GreedyMatchingSpec in rules
    assert len(rules) == 8
