"""Test the `_paths` module of `skcomplex.core.components.laws.utils`."""

import warnings
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction, MapInteraction, TopologyInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import neighborhoods_in, shortest_paths
from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.problems._neighborhood_kernel import NeighborhoodKernel
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import BackendWarning, ValidationError
from skcomplex.core.specs import RoleSpec, StateSpec


class ValueElement(Element):
    """Node or link with one value."""

    def __init__(self, label: str, *, a: Annotated[object, StateSpec()] = 0.0) -> None:
        super().__init__(label)
        self.a = a


class PathsInteraction(Interaction):
    """Drift of every node by a reduction over the shortest paths of the open links."""

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        links: Annotated[str, RoleSpec()],
        reduce: str = 'betweenness',
    ) -> None:
        super().__init__(label)
        self.nodes = nodes
        self.links = links
        self.reduce = reduce

    def _get_laws(self, s):
        """Return the drift of the reduction over the links whose value is not 0."""
        return [RateLaw(s.nodes.a, shortest_paths(s.links.a, reduce=self.reduce))]


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
    system = System(
        elements=[
            GroupElement('nodes', element=ValueElement, space=space, on='sites'),
            GroupElement('links', element=ValueElement, space=space, on='edges', values={'a': weights}),
        ],
        interactions=[PathsInteraction('paths', nodes='nodes', links='links', reduce=reduce)],
    )
    problem = PassCompiler(backend=NumpyBackend()).prepare(system).run().problem_
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
    first = kernel._cached_paths(np.array([0, 1, 2]), np.array([1, 2, 3]))
    assert kernel._cached_paths(np.array([0, 1, 2]), np.array([1, 2, 3])) is first
    assert kernel._cached_paths(np.array([0, 1]), np.array([1, 2])).tolist() == [0.0, 1.0, 0.0, 0.0]


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
def test_shortest_paths_follow_a_network_whose_nodes_are_removed(backend):
    """Test that the load read at every step is the betweenness of the network present, on both backends."""
    graph = nx.barabasi_albert_graph(60, 2, seed=3)
    nodes = GroupElement('nodes', element=ValueElement, on='sites', variable=True, capacity=60)
    links = GroupElement('links', element=Element, on='edges', variable=True, capacity=graph.number_of_edges())
    load = MapInteraction('load', target='nodes', state='a', function=lambda s: shortest_paths(1))
    removal = TopologyInteraction('removal', target='nodes', action='remove-node', probability=0.05)
    space = NetworkSpace(graph)
    system = System(elements=[nodes, links], interactions=[load, removal], space=space, time=DiscreteTime())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BackendWarning)
        trajectory = simulate(system, until=6, backend=backend, random_state=0)
    for frame in range(5):
        expected = nx.betweenness_centrality(trajectory.network(frame).graph(), normalized=False)
        kept = trajectory.network(frame + 1).nodes
        values = trajectory.values('nodes.a')[frame + 1]
        np.testing.assert_allclose(values[kept], [expected[node] for node in kept.tolist()])


def test_numba_runs_shortest_paths_in_python_with_a_warning():
    """Test that the Numba backend warns that it runs the laws of a shortest paths reduction in Python."""
    space = NetworkSpace(nx.path_graph(4))
    system = System(
        elements=[
            GroupElement('nodes', element=ValueElement, space=space, on='sites'),
            GroupElement('links', element=ValueElement, space=space, on='edges', values={'a': 1.0}),
        ],
        interactions=[PathsInteraction('paths', nodes='nodes', links='links')],
    )
    with pytest.warns(BackendWarning, match='shortest paths'):
        problem = PassCompiler(backend=NumbaBackend()).prepare(system).run().problem_
    drift = problem.states.read(problem.drift(0.0, problem.initial_state, problem.initial_parameters), 'nodes.a')
    assert drift.tolist() == [0.0, 2.0, 2.0, 0.0]
