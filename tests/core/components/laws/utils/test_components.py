"""Test the `_components` module of `skcomplex.core.components.laws.utils`."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import both, components
from skcomplex.core.components.laws.utils._reductions import neighborhoods_in
from skcomplex.core.components.spaces import BaseSpace, NetworkSpace
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import Role
from skcomplex.core.systems import System

BACKENDS = pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
SEEDS = pytest.mark.parametrize('seed', [0, 1, 2])


class ComponentsInteraction(Interaction):
    """Drift of every node by a reduction over the components of the active links."""

    def __init__(
        self,
        label: str,
        *,
        nodes: Role,
        links: Role,
        reduce: str = 'label',
        sites: bool = False,
    ) -> None:
        super().__init__(label)
        self.nodes = nodes
        self.links = links
        self.reduce = reduce
        self.sites = sites

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the drift of the reduction over the links that are open, or open between occupied nodes."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            active = s.links.a
            if self.sites:
                active = both(s.links.a != 0, both(s.nodes.a > 0.5, s.nodes.other.a > 0.5))
            self.laws_ = [RateLaw(s.nodes.a, components(active, reduce=self.reduce))]


def _random_network(seed, directed=False):
    """Return a random graph, the weight of every edge, 0 for a closed one, and the occupation of every node."""
    rng = np.random.default_rng(seed)
    graph = nx.gnp_random_graph(40, 0.06, seed=seed, directed=directed)
    size = graph.number_of_edges()
    weights = np.where(rng.uniform(size=size) < 0.7, rng.uniform(0.5, 2.0, size=size), 0.0)
    occupied = (rng.uniform(size=40) < 0.8).astype(float)
    return graph, weights, occupied


def _expected(graph, weights, occupied, reduce, sites):
    """Return the reduction of every node over the components of networkx's graph of the active edges."""
    active = nx.Graph()
    active.add_nodes_from(graph)
    for (first, second), weight in zip(graph.edges, weights, strict=True):
        if weight != 0 and (not sites or (occupied[first] > 0.5 and occupied[second] > 0.5)):
            active.add_edge(first, second)
    expected = np.zeros(len(graph))
    largest = max(len(item) for item in nx.connected_components(active))
    giant = min(min(item) for item in nx.connected_components(active) if len(item) == largest)
    for component in nx.connected_components(active):
        values = {'label': min(component), 'size': len(component), 'largest': largest, 'giant': giant in component}
        expected[list(component)] = values[reduce]
    return expected


def _drift(graph, weights, occupied, backend, reduce, *, sites=False):
    """Return the drift of the nodes, the reduction over the components of the active links."""
    space = NetworkSpace(graph)
    system = System(
        elements=[
            Element('nodes', values={'a': occupied}, space=space),
            Element('links', values={'a': weights}, space=space, on='edges'),
        ],
        interactions=[ComponentsInteraction('components', nodes='nodes', links='links', reduce=reduce, sites=sites)],
        backend=backend,
    )
    problem = system.prepare()._problem
    return problem.states.read(problem.drift(0.0, problem.initial_state, problem.initial_parameters), 'nodes.a')


def test_components_prints_its_reduction():
    """Test that a components reduction prints its activity and reduction."""
    assert str(components(var('a') > 0, reduce='largest')) == "components(a > 0, reduce='largest')"


def test_neighborhoods_in_finds_the_components_reductions():
    """Test that the components reductions of an expression are found with their kinds and activities."""
    a = var('a')
    (found,) = neighborhoods_in(1 + components(a, reduce='size'))
    assert found.kind == 'component-size'
    assert found.argument == a
    assert found.over is None


def test_components_rejects_an_unknown_reduction():
    """Test that an unknown reduction raises an error naming the known ones."""
    with pytest.raises(ValidationError, match='label, size, largest, giant'):
        components(var('a'), reduce='sum')


@BACKENDS
@SEEDS
@pytest.mark.parametrize('reduce', ['label', 'size', 'largest', 'giant'])
def test_components_of_edge_weights_match_networkx(backend, seed, reduce):
    """Test that the components of the edges of nonzero weight equal those of networkx on random graphs."""
    graph, weights, occupied = _random_network(seed)
    drift = _drift(graph, weights, occupied, backend, reduce)
    assert np.array_equal(drift, _expected(graph, weights, occupied, reduce, sites=False))


@BACKENDS
@SEEDS
def test_components_of_a_condition_on_both_ends_match_networkx(backend, seed):
    """Test that links active by a condition on the edge and both of its nodes give networkx's component sizes."""
    graph, weights, occupied = _random_network(seed)
    drift = _drift(graph, weights, occupied, backend, 'size', sites=True)
    assert np.array_equal(drift, _expected(graph, weights, occupied, 'size', sites=True))


@BACKENDS
def test_components_of_a_directed_network_ignore_the_direction(backend):
    """Test that the components of a directed network are its weakly connected components."""
    graph, weights, occupied = _random_network(3, directed=True)
    drift = _drift(graph, weights, occupied, backend, 'label')
    expected = _expected(graph, weights, occupied, 'label', sites=False)
    assert np.array_equal(drift, expected)
