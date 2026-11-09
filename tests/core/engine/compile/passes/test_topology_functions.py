"""Test variable groups on networks with capacity layouts and the topology requests of update and jump laws."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import fit_law
from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction, NeighborInteraction
from skcomplex.core.components.laws import JumpLaw, UpdateLaw
from skcomplex.core.components.laws.utils import neighbors, total
from skcomplex.core.components.spaces import AdjacencyNeighborhood, DiscreteTime, NetworkSpace, RadiusNeighborhood
from skcomplex.core.components.systems import System
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import CapacityError
from skcomplex.core.specs import LayerSpec, ParameterSpec, RoleSpec, StateSpec, TopologyRequestSpec, VisualSpec


def _degree(node):
    """Return the degree of a candidate node, the weight of preferential attachment."""
    return neighbors(1)


def _healthy(node):
    """Return whether a candidate node is susceptible, the weight of a rewired end."""
    return 1 - node.i


class NodeElement(Element):
    """Node element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        age: Annotated[object, StateSpec()] = 0.0,
        x: Annotated[object, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.age = age
        self.x = x

    def _get_laws(self, s):
        """Return the laws of the node element."""
        return [UpdateLaw(s.age, s.age + 1)]


class LinkElement(Element):
    """Link element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        w: Annotated[object, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.w = w


class GrowthElement(Element):
    """Growth element."""

    def _get_laws(self, s):
        """Return the laws of the growth element."""
        return [UpdateLaw(TopologyRequestSpec('add-node', edges=2, toward=_degree), 1)]


class SISElement(Element):
    """S i s element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        i: Annotated[object, StateSpec()] = 0.0,
        beta: Annotated[object, ParameterSpec()] = 0.3,
        gamma: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.i = i
        self.beta = beta
        self.gamma = gamma


class RewiringInteraction(NeighborInteraction):
    """Susceptible ends of edges to infected nodes move them to susceptible nodes at a rate."""

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        links: Annotated[str, RoleSpec()],
        w: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.nodes = nodes
        self.links = links
        self.w = w

    def _get_laws(self, s):
        """Return the rewiring of both orientations of a discordant edge."""
        return [
            JumpLaw(s.w * (1 - s.source.i) * s.target.i, {TopologyRequestSpec('rewire', toward=_healthy): 1}),
            JumpLaw(
                s.w * s.source.i * (1 - s.target.i),
                {TopologyRequestSpec('rewire', keep='target', toward=_healthy): 1},
            ),
        ]


def _growing(size, capacity, *, edges=2):
    """Return a network that starts complete on `size` nodes and grows by preferential attachment."""
    space = NetworkSpace(nx.complete_graph(size))
    return System(
        elements=[
            GroupElement('nodes', element=NodeElement, space=space, on='sites', variable=True, capacity=capacity),
            GroupElement(
                'links',
                element=LinkElement,
                space=space,
                on='edges',
                variable=True,
                capacity=edges * capacity + size**2,
            ),
            GrowthElement('growth'),
        ],
        time=DiscreteTime(),
    )


@pytest.mark.slow
def test_preferential_attachment_gives_a_tail_exponent_near_three():
    """Test that Barabási-Albert growth to 10,000 nodes with two edges each gives a degree tail exponent near 3."""
    trajectory = simulate(_growing(3, 10_003), until=10_000, dt=1000.0, random_state=3)
    snapshot = trajectory.network(-1)
    degrees = np.bincount(np.concatenate([snapshot.sources, snapshot.targets]))
    assert snapshot.nodes.size == 10_003
    fit = fit_law(degrees[degrees > 0], minimum=10, discrete=True, bootstrap=0)
    assert abs(fit.exponent - 3) < 0.3


def test_growth_runs_on_numba_and_agrees_with_numpy():
    """Test that a network growing from 10 to 1,000 nodes runs on Numba with its present nodes, as on NumPy."""
    numpy, numba = (
        simulate(_growing(10, 1000), until=990, backend=backend, dt=99.0, random_state=5)
        for backend in (NumpyBackend(), NumbaBackend())
    )
    assert np.allclose(numpy.parameters, numba.parameters, atol=1e-12, equal_nan=True)
    ages = numba.values('nodes.age')
    for frame, row in enumerate(ages):
        present = numba.network(frame).nodes
        assert np.all(np.isfinite(row[present]))
        assert np.all(np.isnan(np.delete(row, present)))
    assert numba.network(-1).nodes.size == 1000
    assert np.allclose(numpy.values('nodes.age'), ages, equal_nan=True)


def test_snapshot_by_frame_holds_the_network_of_that_frame():
    """Test that the snapshot of a frame of a growing network holds the nodes and edges present at that frame."""
    trajectory = simulate(_growing(3, 40), until=30, dt=1.0, random_state=0)
    snapshot = trajectory.network(10)
    assert snapshot.nodes.tolist() == list(range(13))
    assert snapshot.sources.size == 3 + 2 * 10
    graph = snapshot.graph()
    assert nx.is_connected(graph)
    assert graph.number_of_nodes() == 13


def test_a_full_capacity_raises_a_capacity_error():
    """Test that a request beyond the capacity of a variable group names the group and the fix."""
    with pytest.raises(CapacityError, match='capacity'):
        simulate(_growing(3, 5), until=5, dt=1.0, random_state=0)


def test_requests_remove_and_add_edges_and_nodes():
    """Test that update laws of nodes and edges remove edges and nodes, and add edges to present nodes only."""
    space = NetworkSpace(nx.path_graph(6))

    class PruningInteraction(NeighborInteraction):
        """Pruning interaction."""

        def __init__(
            self,
            label: str,
            *,
            nodes: Annotated[object, RoleSpec()],
            links: Annotated[object, RoleSpec()],
        ) -> None:
            super().__init__(label)
            self.nodes = nodes
            self.links = links

        def _get_laws(self, s):
            """Return the laws of the pruning interaction."""
            return [UpdateLaw(TopologyRequestSpec('remove-edge'), s.source.x * s.target.x)]

    class LeavingElement(Element):
        """Leaving element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            age: Annotated[object, StateSpec()] = 0.0,
            x: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.age = age
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the leaving element."""
            return [UpdateLaw(TopologyRequestSpec('remove-node'), s.x)]

    system = System(
        elements=[
            GroupElement(
                'nodes',
                element=LeavingElement,
                space=space,
                on='sites',
                variable=True,
                values={'x': [1, 1, 0, 0, 0, 1]},
            ),
            GroupElement('links', element=LinkElement, space=space, on='edges', variable=True),
        ],
        interactions=[PruningInteraction('pruning', nodes='nodes', links='links')],
        time=DiscreteTime(),
    )
    trajectory = simulate(system, until=1, dt=1.0, random_state=0)
    snapshot = trajectory.network(1)
    assert snapshot.nodes.tolist() == [2, 3, 4]
    assert sorted(zip(snapshot.sources.tolist(), snapshot.targets.tolist(), strict=True)) == [(2, 3), (3, 4)]

    class JoiningElement(Element):
        """Joining element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the joining element."""
            return [UpdateLaw(TopologyRequestSpec('add-edge', values=(('w', 5.0),)), s.x)]

    space = NetworkSpace(nx.empty_graph(4))
    system = System(
        elements=[
            GroupElement(
                'nodes',
                element=JoiningElement,
                space=space,
                on='sites',
                variable=True,
                values={'x': [1, 0, 0, 0]},
            ),
            GroupElement('links', element=LinkElement, space=space, on='edges', variable=True, capacity=10),
        ],
        time=DiscreteTime(),
    )
    trajectory = simulate(system, until=3, dt=1.0, random_state=0)
    snapshot = trajectory.network(-1)
    assert np.all(snapshot.sources == 0)
    assert sorted(snapshot.targets.tolist()) == [1, 2, 3]
    assert np.allclose(trajectory.values('links.w')[-1][snapshot.edges], 5.0)


def _prevalence(w, seed):
    """Return the mean prevalence of SIS with rewiring at rate `w` late in a run on a random graph."""
    space = NetworkSpace(nx.gnp_random_graph(200, 0.04, seed=seed))
    infected = np.random.default_rng(seed).random(200) < 0.2

    class SpreadInteraction(Interaction):
        """Spread interaction."""

        def __init__(
            self,
            label: str,
            *,
            nodes: Annotated[object, RoleSpec()],
        ) -> None:
            super().__init__(label)
            self.nodes = nodes

        def _get_laws(self, s):
            """Return the laws of the spread interaction."""
            return [
                JumpLaw(s.nodes.beta * (1 - s.nodes.i) * neighbors(s.nodes.other.i), {s.nodes.i: 1}),
                JumpLaw(s.nodes.gamma * s.nodes.i, {s.nodes.i: -1}),
            ]

    system = System(
        elements=[
            GroupElement(
                'nodes',
                element=SISElement,
                space=space,
                on='sites',
                variable=True,
                values={'i': infected.astype(float)},
            ),
            GroupElement('links', element=LinkElement, space=space, on='edges', variable=True),
        ],
        interactions=[
            SpreadInteraction('spread', nodes='nodes'),
            RewiringInteraction('rewiring', nodes='nodes', links='links', w=w),
        ],
    )
    trajectory = simulate(system, until=30, dt=1.0, random_state=seed)
    return float(np.nanmean(trajectory.values('nodes.i')[15:]))


def test_rewiring_raises_the_epidemic_threshold():
    """Test that susceptible nodes rewiring away from infected neighbours suppress an epidemic above its threshold."""
    static, adaptive = _prevalence(0.0, 1), _prevalence(5.0, 1)
    assert static > 0.2
    assert adaptive < 0.5 * static


def test_reductions_over_capacity_groups_count_present_members():
    """Test that a total over a variable group sums the present members only."""

    class CounterElement(Element):
        """Counter element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            size: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.size = size

    class CountingInteraction(Interaction):
        """Counting interaction."""

        def __init__(
            self,
            label: str,
            *,
            counter: Annotated[object, RoleSpec()],
            nodes: Annotated[object, RoleSpec()],
        ) -> None:
            super().__init__(label)
            self.counter = counter
            self.nodes = nodes

        def _get_laws(self, s):
            """Return the laws of the counting interaction."""
            return [UpdateLaw(s.counter.size, total(s.nodes.x))]

    space = NetworkSpace(nx.complete_graph(3))
    system = System(
        elements=[
            GroupElement('nodes', element=NodeElement, space=space, on='sites', variable=True, capacity=10),
            GroupElement('links', element=LinkElement, space=space, on='edges', variable=True, capacity=30),
            GrowthElement('growth'),
            CounterElement('counter'),
        ],
        interactions=[CountingInteraction('counting', counter='counter', nodes='nodes')],
        time=DiscreteTime(),
    )
    trajectory = simulate(system, until=4, dt=1.0, random_state=0)
    assert trajectory.values('counter.size').tolist() == [0.0, 3.0, 4.0, 5.0, 6.0]
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.topology is not None


def test_validation_reports_a_request_on_a_fixed_group():
    """Test that a law that requests a new edge on a network with a fixed edge group gets the issue group-fixed."""
    space = NetworkSpace(nx.path_graph(4))

    class JoiningElement(Element):
        """Joining element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the joining element."""
            return [UpdateLaw(TopologyRequestSpec('add-edge'), 0.5)]

    system = System(
        elements=[
            GroupElement('nodes', element=JoiningElement, space=space, on='sites', variable=True),
            GroupElement('links', element=LinkElement, space=space, on='edges'),
        ],
        time=DiscreteTime(),
    )
    issues = [issue for issue in PassCompiler().prepare(system).report_.issues if issue.code == 'group-fixed']
    assert len(issues) == 1
    assert 'links' in issues[0].message
    assert 'variable=True' in issues[0].fix


def test_validation_reports_a_node_request_on_fixed_site_groups():
    """Test that a node request on a network whose site groups are fixed names the fixed groups."""
    space = NetworkSpace(nx.path_graph(4))
    system = System(
        elements=[
            GroupElement('nodes', element=NodeElement, space=space, on='sites'),
            GroupElement('links', element=LinkElement, space=space, on='edges', variable=True),
            GrowthElement('growth'),
        ],
        time=DiscreteTime(),
    )
    issues = [issue for issue in PassCompiler().prepare(system).report_.issues if issue.code == 'group-fixed']
    assert [issue.path for issue in issues] == ['growth']
    assert 'nodes' in issues[0].message


def test_validation_reports_a_metric_neighbourhood_on_a_network():
    """Test that a reduction over a radius on a network gets law-space with a fix that names a layer."""

    class ReadingElement(Element):
        """Reading element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the reading element."""
            return [UpdateLaw(s.x, neighbors(s.other.x, over=RadiusNeighborhood(1.0)))]

    system = System(
        elements=[GroupElement('nodes', element=ReadingElement, space=NetworkSpace(nx.path_graph(3)), on='sites')],
        time=DiscreteTime(),
    )
    issues = PassCompiler().prepare(system).report_.issues
    assert [issue.code for issue in issues] == ['law-space']
    assert 'continuous space' in issues[0].fix
    assert 'layer' in issues[0].fix


def test_validation_reports_a_reduction_over_a_missing_layer():
    """Test that a reduction over a layer the network lacks gets a binding issue that lists its layers."""

    class ReadingElement(Element):
        """Reading element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the reading element."""
            return [UpdateLaw(s.x, neighbors(s.other.x, over=AdjacencyNeighborhood(layer='trade')))]

    space = NetworkSpace(nx.path_graph(3), layers=[LayerSpec('work', nx.path_graph(3))])
    system = System(
        elements=[GroupElement('nodes', element=ReadingElement, space=space, on='sites')],
        time=DiscreteTime(),
    )
    report = PassCompiler().prepare(system).report_
    issues = report.issues
    assert [issue.code for issue in issues] == ['binding']
    assert "'work'" in issues[0].message
