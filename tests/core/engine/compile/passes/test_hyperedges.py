"""Test the `_hyperedges` module of `skcomplex.core.engine.compile.passes`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import HyperedgeInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import maximum, mean, minimum, total, variance
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.compile.passes._hyperedges import HyperedgeTables
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import RoleSpec, StateSpec

HYPEREDGES = [(0, 1, 2), (1, 2, 3, 4), (2, 4)]
REDUCTIONS = {
    'sum': (lambda s: total(s.members.x, over=s.members), np.sum),
    'mean': (lambda s: mean(s.members.x, over=s.members), np.mean),
    'max': (lambda s: maximum(s.members.x, over=s.members), np.max),
    'min': (lambda s: minimum(s.members.x, over=s.members), np.min),
    'where': (lambda s: total(s.members.x, over=s.members, where=s.members.x > 0.5), lambda v: v[v > 0.5].sum()),
}


class NodeElement(Element):
    """Node with one value."""

    def __init__(self, label: str, *, x: Annotated[object, StateSpec()] = 0.0) -> None:
        super().__init__(label)
        self.x = x


class ReducedInteraction(HyperedgeInteraction):
    """Rate of the focal member given by a reduction over the members of its hyperedge."""

    def __init__(self, label: str, nodes: Annotated[str, RoleSpec()], reduce: str = 'sum') -> None:
        super().__init__(label)
        self.nodes = nodes
        self.reduce = reduce

    def _get_laws(self, s):
        """Return the reduction as the rate of the focal member."""
        if self.reduce == 'variance':
            return [RateLaw(s.member.x, variance(s.members.x, over=s.members))]
        return [RateLaw(s.member.x, REDUCTIONS[self.reduce][0](s))]


def _system(reduce: str, values: np.ndarray) -> CompositeSystem:
    """Return the nodes of a hypergraph whose rates are a reduction over their hyperedges."""
    space = NetworkSpace(nx.complete_graph(5), hyperedges=HYPEREDGES)
    return CompositeSystem(
        elements=[GroupElement('nodes', element=NodeElement, space=space, values={'x': values}, on='sites')],
        interactions=[ReducedInteraction('reduced', nodes='nodes', reduce=reduce)],
    )


@pytest.mark.parametrize('reduce', list(REDUCTIONS))
def test_reductions_over_the_members_are_written_out_per_hyperedge(reduce):
    """Test that every reduction over the members of a hyperedge equals the reduction of its member values."""
    values = np.array([0.1, 0.9, 0.4, 0.7, 0.6])
    problem = PassCompiler().prepare(_system(reduce, values)).run().problem_
    drift = problem.states.read(problem.drift(0.0, problem.initial_state, problem.initial_parameters), 'nodes.x')
    expected = np.zeros(5)
    for hyperedge in HYPEREDGES:
        expected[list(hyperedge)] += REDUCTIONS[reduce][1](values[list(hyperedge)])
    assert np.allclose(drift, expected, atol=1e-12)


def test_variance_over_the_members_is_rejected():
    """Test that a reduction kind a hyperedge does not write out raises an error naming the kinds it takes."""
    with pytest.raises(ValidationError, match='total, mean, count, maximum, or minimum'):
        PassCompiler().prepare(_system('variance', np.zeros(5))).run()


def test_hyperedge_tables_get_incidences():
    """Test `HyperedgeTables._get_incidences` on a small case."""
    space = NetworkSpace(nx.complete_graph(4), hyperedges=[(0, 1, 2), (1, 2, 3)]).prepare()
    indptr, others = HyperedgeTables._get_incidences(space, 3)
    assert (indptr.tolist(), [item.tolist() for item in others]) == (
        [0, 1, 3, 5, 6],
        [[1, 2, 0, 1, 0, 1], [2, 3, 2, 3, 1, 2]],
    )
