"""Test the `_paths` module of `skcomplex.core.components.laws.utils`."""

import warnings
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, TopologyRequestSpec, UpdateLaw
from skcomplex.core.components.laws.utils import shortest_paths
from skcomplex.core.components.laws.utils._reductions import neighborhoods_in
from skcomplex.core.components.spaces import BaseSpace, NetworkSpace
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.problems._neighborhood_kernel import NeighborhoodKernel
from skcomplex.core.errors import BackendWarning, ValidationError
from skcomplex.core.specs import Role
from skcomplex.core.systems import System


class PathsInteraction(Interaction):
    """Drift of every node by a reduction over the shortest paths of the open links."""

    def __init__(
        self,
        label: str,
        *,
        nodes: Role,
        links: Role,
        reduce: str = 'betweenness',
    ) -> None:
        super().__init__(label)
        self.nodes = nodes
        self.links = links
        self.reduce = reduce

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the drift of the reduction over the links whose value is not 0."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [RateLaw(context.nodes.a, shortest_paths(context.links.a, reduce=self.reduce))]


def _paths(space, weights, reduce='betweenness', backend='auto'):
    """Return the nodes and links of a network whose nodes drift by a reduction over the shortest paths."""
    return System(
        elements=[
            Element('nodes', values={'a': 0.0}, space=space),
            Element('links', values={'a': weights}, space=space, on='edges'),
        ],
        interactions=[PathsInteraction('paths', nodes='nodes', links='links', reduce=reduce)],
        backend=backend,
    )


def test_shortest_paths_prints_and_lists_its_kind():
    """Test that the reduction prints with its keywords and is listed as a neighbourhood reduction of its kind."""
    assert str(shortest_paths(var('up'), reduce='closeness')) == "shortest_paths(up, reduce='closeness')"
    assert [item.kind for item in neighborhoods_in(1 + shortest_paths(1))] == ['path-betweenness']


def test_shortest_paths_rejects_an_unknown_reduction():
    """Test that an unknown reduction raises an error."""
    with pytest.raises(ValidationError, match='Unknown shortest paths'):
        shortest_paths(1, reduce='diameter')


@pytest.mark.parametrize('reduce', ['betweenness', 'closeness'])
def test_shortest_paths_agrees_with_networkx_on_the_open_links(reduce):
    """Test that the drift equals NetworkX's measure on the graph of the links whose value is not 0."""
    rng = np.random.default_rng(0)
    graph = nx.gnp_random_graph(30, 0.15, seed=1)
    weights = (rng.uniform(size=graph.number_of_edges()) < 0.8).astype(float)
    space = NetworkSpace(graph)
    problem = _paths(space, weights, reduce, NumpyBackend()).prepare()._problem
    drift = problem.states.read(problem.drift(0.0, problem.initial_state, problem.initial_parameters), 'nodes.a')
    open_graph = nx.Graph()
    open_graph.add_nodes_from(graph)
    open_graph.add_edges_from(edge for edge, weight in zip(graph.edges, weights, strict=True) if weight)
    measure = nx.betweenness_centrality(open_graph, normalized=False) if reduce == 'betweenness' else None
    measure = nx.closeness_centrality(open_graph) if measure is None else measure
    np.testing.assert_allclose(drift, [measure[node] for node in graph.nodes])


def test_neighborhood_kernel_reuses_the_paths_of_unchanged_links():
    """Test that the kernel keeps the values of the last links and computes them again only for other links."""
    kernel = NeighborhoodKernel.__new__(NeighborhoodKernel)
    kernel.kind, kernel.paths_ = 'path-betweenness', None
    kernel.entry = type('Entry', (), {'rows': 4})()
    first = kernel._get_cached_paths(np.array([0, 1, 2]), np.array([1, 2, 3]))
    assert kernel._get_cached_paths(np.array([0, 1, 2]), np.array([1, 2, 3])) is first
    assert kernel._get_cached_paths(np.array([0, 1]), np.array([1, 2])).tolist() == [0.0, 1.0, 0.0, 0.0]


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
def test_shortest_paths_follow_a_network_whose_nodes_are_removed(backend):
    """Test that the load read at every step is the betweenness of the network present, on both backends."""
    graph = nx.barabasi_albert_graph(60, 2, seed=3)
    nodes = Element('nodes', values={'a': 0.0}, space=NetworkSpace(graph))
    load = Interaction('load', roles={'target': 'nodes'}, laws=lambda s: [UpdateLaw(s.target.a, shortest_paths(1))])
    removal = Interaction(
        'removal',
        roles={'target': 'nodes'},
        laws=lambda s: [UpdateLaw(TopologyRequestSpec('remove-node'), 0.05)],
    )
    system = System(elements=[nodes], interactions=[load, removal], backend=backend)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BackendWarning)
        trajectory = system.prepare().simulate(until=6, random_state=0)
    for frame in range(5):
        expected = nx.betweenness_centrality(trajectory.get_network(frame).get_graph(), normalized=False)
        kept = trajectory.get_network(frame + 1).nodes
        values = trajectory.get_values('nodes.a')[frame + 1]
        np.testing.assert_allclose(values[kept], [expected[node] for node in kept.tolist()])


def test_numba_runs_shortest_paths_in_python_with_a_warning():
    """Test that the Numba backend warns that it runs the laws of a shortest paths reduction in Python."""
    space = NetworkSpace(nx.path_graph(4))
    with pytest.warns(BackendWarning, match='shortest paths'):
        problem = _paths(space, 1.0, backend=NumbaBackend()).prepare()._problem
    drift = problem.states.read(problem.drift(0.0, problem.initial_state, problem.initial_parameters), 'nodes.a')
    assert drift.tolist() == [0.0, 2.0, 2.0, 0.0]
