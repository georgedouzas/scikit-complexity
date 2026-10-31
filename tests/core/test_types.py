"""Test the `types` module of `skcomplex.core`."""

import typing
from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from skcomplex.core.types import AgentViewProtocol, Preference, Process, Target


def test_targets_are_symbolic_expressions_or_vectors():
    """Test that a law targets a symbolic expression or a vector of them."""
    assert typing.get_args(Target.__value__) == (Expression, FreeModuleElement)


def test_processes_may_be_nothing():
    """Test that a process may be nothing."""
    assert type(None) in typing.get_args(Process.__value__)


def test_agent_views_have_an_id_a_location_and_a_life():
    """Test that an object with an id, a location, and the moves of an agent is an agent view, and one without is not."""
    names = ('neighbor_cells', 'move_to', 'remove', 'create')
    view = SimpleNamespace(id=1, location=(0, 0), alive=True, **dict.fromkeys(names, print))
    assert isinstance(view, AgentViewProtocol)
    assert not isinstance(SimpleNamespace(id=1), AgentViewProtocol)


def test_preferences_score_two_agents():
    """Test that a preference is a function of two agents that returns a score."""
    assert typing.get_args(Preference.__value__)[-1] is float
