"""Test the `_hyperedge` module of `skcomplex.core.components.interactions`."""

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import sin

from skcomplex.core.components.elements import Element, GroupElement, LatticeElement
from skcomplex.core.components.interactions import HyperedgeInteraction
from skcomplex.core.components.laws import RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import count, maximum, mean, total
from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile import PassCompiler, compile_system
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import RoleSpec, StateSpec, VisualSpec

BACKENDS = pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
HYPEREDGES = [(0, 1, 2), (1, 2, 3, 4), (3, 4, 5), (0, 2, 4, 5), (2, 5)]


class NodeElement(Element):
    """Node with one value."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x


class LinkElement(Element):
    """Link or hyperedge with an activity."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a = a


class ConsensusInteraction(HyperedgeInteraction):
    """Interaction over the hyperedges of the voters."""

    def __init__(self, label: str, *, voters: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label)
        self.voters = voters


class PullInteraction(HyperedgeInteraction):
    """Pull of every member towards the members of its hyperedges."""

    def __init__(self, label: str, nodes: Annotated[str, RoleSpec()], size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.nodes = nodes

    def _get_laws(self, s):
        """Return the pull of the focal member."""
        return [RateLaw(s.member.x, total(s.members.x, over=s.members) - count(over=s.members) * s.member.x)]


class TriadInteraction(HyperedgeInteraction):
    """Three-body phase coupling of the members of the triangles."""

    def __init__(self, label: str, nodes: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, size=3)
        self.nodes = nodes

    def _get_laws(self, s):
        """Return the coupling of the focal member to the other two."""
        return [RateLaw(s.member.x, sin(s.others[0].x + s.others[1].x - 2 * s.member.x))]


class RecordInteraction(HyperedgeInteraction):
    """Activity of a hyperedge or a link set from its members."""

    def __init__(
        self,
        label: str,
        nodes: Annotated[str, RoleSpec()],
        links: Annotated[str, RoleSpec()],
        size: int | None = None,
        part: str = 'hyperedge',
    ) -> None:
        super().__init__(label, size=size)
        self.nodes = nodes
        self.links = links
        self.part = part

    def _get_laws(self, s):
        """Return the update of the activity of the hyperedge or of the link."""
        target = getattr(s, self.part).a
        if self.part == 'edge':
            return [UpdateLaw(target, s.others[-1].x)]
        return [UpdateLaw(target, maximum(s.members.x, over=s.members) + mean(s.members.x, over=s.members))]


def _drift(interaction, backend, values):
    """Return the drift of the nodes of the network of the hyperedges coupled by an interaction."""
    space = NetworkSpace(nx.complete_graph(6), hyperedges=HYPEREDGES)
    system = System(
        elements=[LatticeElement('nodes', element=NodeElement, space=space, values={'x': values})],
        interactions=[interaction],
    )
    problem = compile_system(system, backend=backend)
    return problem.states.read(problem.drift(0.0, problem.initial_state, problem.initial_parameters), 'nodes.x')


def test_hyperedge_interaction_has_no_laws_by_default():
    """Test that a hyperedge interaction stores its role and runs into no laws of its own."""
    consensus = ConsensusInteraction('consensus', voters='voters').prepare()
    assert consensus.voters == 'voters'
    assert consensus.size is None
    assert list(ConsensusInteraction._get_specs()) == ['voters']
    assert consensus.run(SimpleNamespace()).laws_ == []


@pytest.mark.parametrize('size', [1, 2.5, True])
def test_hyperedge_interaction_rejects_a_size_below_two(size):
    """Test that a size that is not a whole number of at least two raises an error."""
    with pytest.raises(ValidationError, match='whole number of 2 or more'):
        PullInteraction('pull', nodes='nodes', size=size).prepare()


@BACKENDS
def test_hyperedge_law_matches_a_loop_over_hyperedges_of_every_size(backend):
    """Test that a law of the focal member over hyperedges of sizes 2, 3, and 4 equals a loop over the hyperedges."""
    values = np.random.default_rng(0).normal(size=6)
    expected = np.zeros(6)
    for hyperedge in HYPEREDGES:
        for member in hyperedge:
            expected[member] += values[list(hyperedge)].sum() - len(hyperedge) * values[member]
    drift = _drift(PullInteraction('pull', nodes='nodes'), backend, values)
    assert np.allclose(drift, expected, rtol=0.0, atol=1e-12)


@BACKENDS
def test_hyperedge_law_of_one_size_reads_the_other_members_in_order(backend):
    """Test that a three-body coupling over the triangles equals a loop over the triangles."""
    values = np.random.default_rng(1).uniform(0.0, 2 * np.pi, size=6)
    expected = np.zeros(6)
    for hyperedge in (item for item in HYPEREDGES if len(item) == 3):
        for member in hyperedge:
            first, second = (other for other in hyperedge if other != member)
            expected[member] += np.sin(values[first] + values[second] - 2 * values[member])
    drift = _drift(TriadInteraction('triad', nodes='nodes'), backend, values)
    assert np.allclose(drift, expected, rtol=0.0, atol=1e-12)


def test_hyperedge_interaction_of_a_missing_size_is_reported():
    """Test that a size no hyperedge of the network has gives a binding issue naming both sizes."""
    space = NetworkSpace(nx.complete_graph(4), hyperedges=[(0, 1, 2), (1, 2, 3)])
    system = System(
        elements=[LatticeElement('nodes', element=NodeElement, space=space)],
        interactions=[PullInteraction('pull', nodes='nodes', size=4)],
    )
    issues = [issue for issue in PassCompiler().prepare(system).report_.issues if issue.code == 'binding']
    (issue,) = issues
    assert 'size 4' in issue.message
    assert 'size 3' in issue.message
    with pytest.raises(ValidationError, match='size 4'):
        compile_system(system)


def test_hyperedge_group_takes_an_update_once_per_hyperedge():
    """Test that an update of a group on hyperedges runs once per hyperedge of the size, leaving the others."""
    space = NetworkSpace(nx.complete_graph(6), hyperedges=HYPEREDGES)
    values = np.arange(6, dtype=float)
    system = System(
        elements=[
            LatticeElement('nodes', element=NodeElement, space=space, values={'x': values}),
            GroupElement('groups', element=LinkElement, space=space, on='hyperedges', values={'a': -1.0}),
        ],
        interactions=[RecordInteraction('record', nodes='nodes', links='groups', size=3)],
        time=DiscreteTime(),
    )
    simulation = Simulation(system, dt=1).prepare().run(1)
    activities = simulation.trajectory_.values('groups.a')[-1]
    expected = [values[list(item)].max() + values[list(item)].mean() if len(item) == 3 else -1.0 for item in HYPEREDGES]
    assert np.allclose(activities, expected, atol=1e-12)


def test_edge_group_takes_the_update_of_the_regulating_member():
    """Test that the edge joining the first two members of every triangle takes the value of its third member."""
    graph = nx.path_graph(4)
    space = NetworkSpace(graph, hyperedges=[(0, 1, 3), (2, 3, 0)])
    values = np.array([0.0, 1.0, 2.0, 3.0])
    system = System(
        elements=[
            LatticeElement('nodes', element=NodeElement, space=space, values={'x': values}),
            GroupElement('links', element=LinkElement, space=space, on='edges', values={'a': -1.0}),
        ],
        interactions=[RecordInteraction('record', nodes='nodes', links='links', part='edge')],
        time=DiscreteTime(),
    )
    simulation = Simulation(system, dt=1).prepare().run(1)
    assert simulation.trajectory_.values('links.a')[-1].tolist() == [3.0, -1.0, 0.0]


def test_edge_group_of_a_hyperedge_without_an_edge_is_reported():
    """Test that an edge role whose hyperedge has its first two members unjoined gives a binding issue."""
    space = NetworkSpace(nx.path_graph(4), hyperedges=[(0, 2, 1)])
    system = System(
        elements=[
            LatticeElement('nodes', element=NodeElement, space=space),
            GroupElement('links', element=LinkElement, space=space, on='edges'),
        ],
        interactions=[RecordInteraction('record', nodes='nodes', links='links', part='edge')],
        time=DiscreteTime(),
    )
    issues = [issue for issue in PassCompiler().prepare(system).report_.issues if issue.code == 'binding']
    (issue,) = issues
    assert 'no edge between the first two members' in issue.message
    with pytest.raises(ValidationError, match='no edge between'):
        compile_system(system)


def _growing(interaction):
    """Return a network of four nodes whose groups on the nodes and on the triangles are variable."""
    nodes = GroupElement('nodes', element=NodeElement, on='sites', variable=True, capacity=8)
    triangles = GroupElement('triangles', element=LinkElement, on='hyperedges', variable=True, capacity=4)
    space = NetworkSpace(nx.complete_graph(4), hyperedges=[[0, 1, 2]])
    return System(elements=[nodes, triangles], interactions=[interaction], space=space)


@BACKENDS
def test_hyperedge_law_reads_the_hyperedges_of_a_growing_network(backend):
    """Test that the drift follows the hyperedges written into the rows of a variable group, without its free rows."""
    problem = compile_system(_growing(PullInteraction('pull', nodes='nodes')), backend=backend)
    hyperedges = [[0, 1, 2], [1, 2, 3], [0, 2, 3]]
    parameters = np.array(problem.initial_parameters, dtype=float)
    _, members = problem.parameters.locate('triangles._members')
    _, active = problem.parameters.locate('triangles._active')
    parameters[members] = np.ravel([*hyperedges, [5, 5, 5]])
    parameters[active] = [1.0, 1.0, 1.0, 0.0]
    state = np.random.default_rng(1).random(np.size(problem.initial_state))
    values = problem.states.read(state, 'nodes.x')
    expected = np.zeros(8)
    for hyperedge in hyperedges:
        expected[hyperedge] += values[hyperedge].sum() - 3 * values[hyperedge]
    drift = problem.states.read(problem.drift(0.0, state, parameters), 'nodes.x')
    np.testing.assert_allclose(drift, expected, atol=1e-12)


def test_hyperedge_update_of_the_focal_member_of_a_growing_network_is_reported():
    """Test that an update of the focal member on hyperedges that change is reported."""

    class SetInteraction(HyperedgeInteraction):
        """Update of every member to the mean of its hyperedge."""

        def __init__(self, label: str, nodes: Annotated[str, RoleSpec()]) -> None:
            super().__init__(label)
            self.nodes = nodes

        def _get_laws(self, s):
            """Return the update of the focal member."""
            return [UpdateLaw(s.member.x, mean(s.members.x, over=s.members))]

    system = _growing(SetInteraction('set', nodes='nodes'))
    system.time = DiscreteTime()
    with pytest.raises(ValidationError, match='change during a run'):
        Simulation(system).prepare()
