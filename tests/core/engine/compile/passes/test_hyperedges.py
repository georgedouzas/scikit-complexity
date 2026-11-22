"""Test the `_hyperedges` module of `skcomplex.core.engine.compile.passes`."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import maximum, mean, minimum, total, variance
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.engine.compile.passes._hyperedges import HyperedgeTables
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role

HYPEREDGES = [(0, 1, 2), (1, 2, 3, 4), (2, 4)]
REDUCTIONS = {
    'sum': (lambda s: total(s.members.x, over=s.members), np.sum),
    'mean': (lambda s: mean(s.members.x, over=s.members), np.mean),
    'max': (lambda s: maximum(s.members.x, over=s.members), np.max),
    'min': (lambda s: minimum(s.members.x, over=s.members), np.min),
    'where': (lambda s: total(s.members.x, over=s.members, where=s.members.x > 0.5), lambda v: v[v > 0.5].sum()),
}


class ReducedInteraction(Interaction):
    """Rate of the focal member given by a reduction over the members of its hyperedge."""

    def __init__(self, label: str, *, among: Role, along: str = 'hyperedges', reduce: str = 'sum') -> None:
        super().__init__(label, among=among, along=along)
        self.reduce = reduce

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the reduction as the rate of the focal member."""
        super()._prepare(context)
        s = context
        if self.reduce == 'variance':
            self.laws_ = [RateLaw(s.member.x, variance(s.members.x, over=s.members))]
        else:
            self.laws_ = [RateLaw(s.member.x, REDUCTIONS[self.reduce][0](s))]


def _system(reduce: str, values: np.ndarray) -> System:
    """Return the nodes of a hypergraph whose rates are a reduction over their hyperedges."""
    space = NetworkSpace(nx.complete_graph(5), hyperedges=HYPEREDGES)
    return System(
        elements=[Element('nodes', values={'x': values}, space=space)],
        interactions=[ReducedInteraction('reduced', among='nodes', reduce=reduce)],
    )


@pytest.mark.parametrize('reduce', list(REDUCTIONS))
def test_reductions_over_the_members_are_written_out_per_hyperedge(reduce):
    """Test that every reduction over the members of a hyperedge equals the reduction of its member values."""
    values = np.array([0.1, 0.9, 0.4, 0.7, 0.6])
    problem = _system(reduce, values).prepare()._problem
    drift = problem.states.read(problem.drift(0.0, problem.initial_state, problem.initial_parameters), 'nodes.x')
    expected = np.zeros(5)
    for hyperedge in HYPEREDGES:
        expected[list(hyperedge)] += REDUCTIONS[reduce][1](values[list(hyperedge)])
    assert np.allclose(drift, expected, atol=1e-12)


def test_variance_over_the_members_is_rejected():
    """Test that a reduction kind a hyperedge does not write out raises an error naming the kinds it takes."""
    with pytest.raises(ValidationError, match='total, mean, count, maximum, or minimum'):
        _system('variance', np.zeros(5)).prepare()


def test_hyperedge_tables_get_incidences():
    """Test `HyperedgeTables._get_incidences` on a small case."""
    space = NetworkSpace(nx.complete_graph(4), hyperedges=[(0, 1, 2), (1, 2, 3)]).prepare()
    indptr, others = HyperedgeTables._get_incidences(space, 3)
    assert (indptr.tolist(), [item.tolist() for item in others]) == (
        [0, 1, 3, 5, 6],
        [[1, 2, 0, 1, 0, 1], [2, 3, 2, 3, 1, 2]],
    )
