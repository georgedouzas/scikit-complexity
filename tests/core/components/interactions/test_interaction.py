"""Test the `_interaction` module of `skcomplex.core.components.interactions`."""

from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet
from scipy.linalg import expm
from sklearn.base import clone

from skcomplex.core.components.elements import BaseElement, Element, GroupElement
from skcomplex.core.components.interactions import Interaction, WienerProcessSpec
from skcomplex.core.components.laws import NoiseLaw, RateLaw, TopologyRequestSpec, UpdateLaw
from skcomplex.core.components.laws.utils import neighbors, total
from skcomplex.core.components.spaces import BaseSpace, ContinuousSpace, LatticeSpace, NetworkSpace, RadiusNeighborhood
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.errors import LabelError, RoleError, ValidationError
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec
from skcomplex.core.systems import System
from skcomplex.core.types import Role

from ....conftest import HoldingElement, _correlated


class StockElement(Element):
    """Stock with one nonnegative state."""

    def __init__(
        self,
        label: str,
        *,
        n: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.n = n


class GrowthInteraction(Interaction):
    """Exponential growth of the state of its target."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.r = r

    def _get_laws(self, s):
        """Return the rate of growth."""
        return [RateLaw(s.target.n, s.r * s.target.n)]


class PredationInteraction(Interaction):
    """Lotka-Volterra predation of prey by a predator."""

    def __init__(
        self,
        label: str,
        *,
        predator: Annotated[str, RoleSpec()],
        prey: Annotated[str, RoleSpec()],
        a: Annotated[float, ParameterSpec()] = 1.0,
        b: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.predator = predator
        self.prey = prey
        self.a = a
        self.b = b

    def _get_laws(self, s):
        """Return the loss of prey and the gain of the predator."""
        return [
            RateLaw(s.prey.n, -s.a * s.predator.n * s.prey.n),
            RateLaw(s.predator.n, s.b * s.predator.n * s.prey.n),
        ]


class TransferInteraction(Interaction):
    """Inflow into a sink at the value of a source."""

    def __init__(
        self,
        label: str,
        *,
        source: Annotated[str, RoleSpec()],
        sink: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.source = source
        self.sink = sink

    def _get_laws(self, s):
        """Return the inflow into the sink."""
        return [RateLaw(s.sink.n, s.source.n)]


class ShockInteraction(Interaction):
    """Unit noise on one element from the first process of a set of Wiener processes."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        noise: Annotated[WienerProcessSpec, RoleSpec(noise=True)],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.noise = noise

    def _get_laws(self, s):
        """Return the noise of the target."""
        return [NoiseLaw(s.target.n, 1.0, process=s.noise.dW)]


class WeightedInteraction(Interaction):
    """Drift of every node by the sum of its neighbours weighted by the states of the edges that join them."""

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        links: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.nodes = nodes
        self.links = links

    def _get_laws(self, s):
        """Return the weighted sum of the neighbours."""
        return [RateLaw(s.nodes.n, neighbors(s.nodes.other.n, weight=s.links.n))]


def _shocked(first, second):
    """Return two holdings, each shocked by an interaction that binds its own Wiener processes."""
    return CompositeSystem(
        elements=[HoldingElement('a'), HoldingElement('b')],
        interactions=[
            ShockInteraction('first', target='a', noise=first),
            ShockInteraction('second', target='b', noise=second),
        ],
    )


def test_interaction_has_no_laws_by_default():
    """Test that an interaction that returns no laws of its own runs into none."""
    assert Interaction('empty').prepare().run(SimpleNamespace()).laws_ == []


def test_interaction_requires_its_roles():
    """Test that an interaction without a required role cannot be made."""
    with pytest.raises(TypeError, match='predator'):
        PredationInteraction('hunt', prey='rabbits')


def test_interaction_clones_and_pickles_with_its_values():
    """Test that an interaction clones and pickles with its roles and parameters."""
    growth = GrowthInteraction('g', target='a', r=0.7)
    assert clone(growth).get_params() == {'label': 'g', 'r': 0.7, 'target': 'a'}
    data = ForkingPickler.dumps(CompositeSystem(elements=[StockElement('a', n=3.0)], interactions=[growth]))
    restored = ForkingPickler.loads(data)
    assert restored.elements[0].n == 3.0
    assert restored.interactions[0].r == 0.7


def test_role_binds_to_group_and_applies_to_every_member():
    """Test that an interaction bound to a group applies its law to every member."""
    system = CompositeSystem(
        elements=[GroupElement('herds', element=StockElement, size=3, values={'n': [1.0, 2.0, 3.0]})],
        interactions=[GrowthInteraction('growth', target='herds', r=2.0)],
    )
    problem = PassCompiler().prepare(system).run().problem_
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert drift.tolist() == [2.0, 4.0, 6.0]


def test_group_of_members_grows_like_one_member():
    """Test that a group of 100 members grows every member at the rate of the interaction bound to it."""
    system = CompositeSystem(
        elements=[GroupElement('herd', element=StockElement, size=100, values={'n': np.linspace(1, 2, 100)})],
        interactions=[GrowthInteraction('g', target='herd')],
    )
    trajectory = simulate(system, until=1, dt=1)
    assert trajectory.get_values('herd.n').shape == (2, 100)
    assert np.allclose(trajectory.get_values('herd.n')[-1], np.linspace(1, 2, 100) * np.exp(0.5), rtol=1e-3)


def test_single_element_bound_with_group_receives_sum():
    """Test that a single element bound next to a group receives the sum over the members."""
    system = CompositeSystem(
        elements=[
            GroupElement('rabbits', element=StockElement, size=3, values={'n': [1.0, 2.0, 3.0]}),
            StockElement('fox', n=2.0),
        ],
        interactions=[PredationInteraction('hunt', predator='fox', prey='rabbits')],
    )
    problem = PassCompiler().prepare(system).run().problem_
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert drift.tolist() == [-2.0, -4.0, -6.0, 12.0]
    assert problem.has_group_sums


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
def test_interaction_reads_two_groups_member_by_member(backend):
    """Test that an interaction bound to two groups of equal size reads row `i` of both together."""
    system = CompositeSystem(
        elements=[
            GroupElement('source', element=StockElement, size=5, values={'n': np.arange(5.0)}),
            GroupElement('sink', element=StockElement, size=5),
        ],
        interactions=[TransferInteraction('flow', source='source', sink='sink')],
    )
    problem = PassCompiler(backend=backend).prepare(system).run().problem_
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert problem.states.read(drift, 'sink.n').tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_interaction_reading_groups_of_different_sizes_raises_error():
    """Test that an interaction reading groups of different sizes member by member names both groups and sizes."""
    system = CompositeSystem(
        elements=[
            GroupElement('source', element=StockElement, size=5),
            GroupElement('sink', element=StockElement, size=7),
        ],
        interactions=[TransferInteraction('flow', source='source', sink='sink')],
    )
    with pytest.raises(RoleError, match=r'sink of 7 members, source of 5 members'):
        PassCompiler().prepare(system).run()


def test_correlated_noise():
    """Test that two noise laws drawing the processes of a set bound to a role have the increments `h C`."""
    covariance = [[1.0, 0.6], [0.6, 2.0]]
    trajectory = simulate(
        _correlated(covariance),
        until=5_000,
        solver=EulerMaruyamaSolver(dt=0.5),
        random_state=0,
        dt=0.5,
    )
    increments = np.diff(np.stack([trajectory.get_values('a.n'), trajectory.get_values('b.n')]), axis=1)
    np.testing.assert_allclose(np.cov(increments), 0.5 * np.asarray(covariance), atol=0.05)
    assert not issubclass(WienerProcessSpec, BaseElement)


def test_interactions_share_processes_of_one_label():
    """Test that two interactions that bind processes of one label draw the same increments."""
    market = WienerProcessSpec('market')
    trajectory = simulate(_shocked(market, WienerProcessSpec('market')), until=1, random_state=0)
    np.testing.assert_array_equal(trajectory.get_values('a.n'), trajectory.get_values('b.n'))
    assert PassCompiler().prepare(_shocked(market, market)).run().problem_.n_processes == 1


def test_processes_of_one_label_must_agree():
    """Test that two different sets of processes with one label, and a path bound to a noise role, are rejected."""
    with pytest.raises(LabelError, match='two different Wiener processes'):
        PassCompiler().prepare(_shocked(WienerProcessSpec('market'), WienerProcessSpec('market', dimension=2))).run()
    with pytest.raises(LabelError, match='label of another child'):
        PassCompiler().prepare(_shocked(WienerProcessSpec('a'), WienerProcessSpec('a'))).run()
    with pytest.raises(RoleError, match='takes a WienerProcessSpec'):
        PassCompiler().prepare(_shocked('market', 'market')).run()


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
def test_edge_states_weigh_neighbourhood_reductions(backend):
    """Test that a reduction over a network weighs every neighbour with the state of the edge that joins them."""
    space = NetworkSpace(nx.path_graph(4))
    system = CompositeSystem(
        elements=[
            GroupElement('nodes', element=StockElement, space=space, values={'n': [1.0, 2.0, 3.0, 4.0]}, on='sites'),
            GroupElement('links', element=StockElement, space=space, on='edges', values={'n': [10.0, 100.0, 1000.0]}),
        ],
        interactions=[WeightedInteraction('weighted', nodes='nodes', links='links')],
    )
    problem = PassCompiler(backend=backend).prepare(system).run().problem_
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    expected = [10.0 * 2, 10.0 * 1 + 100.0 * 3, 100.0 * 2 + 1000.0 * 4, 1000.0 * 3]
    assert np.allclose(problem.states.read(drift, 'nodes.n'), expected, atol=1e-9)


class _ExchangeInteraction(Interaction):
    """Exchange of a value between the two ends of every edge."""

    def __init__(self, label: str, *, among: Role = 'nodes', along: bool | str | None = True, k: float = 1.0) -> None:
        super().__init__(label, among=among, along=along)
        self.k = k

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the flow from the source to the target of an edge."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            flow = context.k * (context.source.x - context.target.x)
            self.laws_ = [
                RateLaw(context.target.x.diff(context.t), flow),
                RateLaw(context.source.x.diff(context.t), -flow),
            ]


class _RewiringInteraction(Interaction):
    """Rewiring that moves an edge away from two voters who disagree."""

    def __init__(self, label: str, *, among: Role = 'voters', along: bool | str | None = True, p: float = 1.0) -> None:
        super().__init__(label, among=among, along=along)
        self.p = p

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rewiring of an edge with the probability that its ends disagree."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            disagreement = (context.source.opinion - context.target.opinion) ** 2
            self.laws_ = [UpdateLaw(TopologyRequestSpec('rewire'), context.p * disagreement)]


def test_interaction_along_edges_runs_once_per_edge():
    """Test that an exchange along the edges of a path of four nodes follows the exact consensus `exp(-L t) x`."""
    graph = nx.path_graph(4)
    start = np.array([1.0, 0.0, 0.0, 0.0])
    nodes = Element('nodes', values={'x': start}, space=NetworkSpace(graph))
    system = System(elements=[nodes], interactions=[_ExchangeInteraction('exchange')]).prepare()
    last = system.simulate(until=0.5, dt=0.01).get_values('nodes.x')[-1]
    assert last == pytest.approx(expm(-nx.laplacian_matrix(graph).toarray() * 0.5) @ start, abs=1e-6)


def test_an_interaction_without_a_class_runs_along_edges():
    """Test that an exchange written without a class, with `among` and `along`, matches the exact consensus."""
    graph = nx.path_graph(4)
    start = np.array([1.0, 0.0, 0.0, 0.0])
    nodes = Element('nodes', values={'x': start}, space=NetworkSpace(graph))
    exchange = Interaction(
        'exchange',
        among='nodes',
        along=True,
        laws=lambda s: [RateLaw(s.target.x, s.source.x - s.target.x), RateLaw(s.source.x, s.target.x - s.source.x)],
    )
    last = System(elements=[nodes], interactions=[exchange]).prepare().simulate(until=0.5, dt=0.01)
    expected = expm(-nx.laplacian_matrix(graph).toarray() * 0.5) @ start
    assert last.get_values('nodes.x')[-1] == pytest.approx(expected, abs=1e-6)


def test_interaction_along_edges_clones_with_its_binding():
    """Test that the binding of an interaction is one of its parameters, kept by a clone."""
    copy = clone(_ExchangeInteraction('exchange', among='cells', along='roads'))
    assert (copy.among, copy.along, copy._get_layer(), copy._runs_on_edges()) == ('cells', 'roads', 'roads', True)
    assert not Interaction('growth')._runs_on_edges()


def test_interaction_along_hyperedges_runs_on_no_edge():
    """Test that `along='hyperedges'` runs the laws once per hyperedge, not per edge, on the graph of no layer."""
    interaction = _ExchangeInteraction('exchange', along='hyperedges')
    assert (interaction._runs_on_hyperedges(), interaction._runs_on_edges(), interaction._get_layer()) == (
        True,
        False,
        None,
    )


def test_interaction_along_hyperedges_of_one_size_skips_the_others():
    """Test that `along=('hyperedges', 3)` runs on the triangle of a network and not on its pair."""
    space = NetworkSpace(nx.empty_graph(4), hyperedges=[[0, 1], [0, 1, 2]])
    nodes = Element('nodes', values={'x': np.zeros(4)}, space=space)
    push = Interaction(
        'push',
        among='nodes',
        along=('hyperedges', 3),
        laws=lambda s: [RateLaw(s.member.x.diff(s.t), 1.0)],
    )
    trajectory = System(elements=[nodes], interactions=[push]).prepare().simulate(until=1.0, dt=0.5)
    assert trajectory.get_values('nodes.x')[-1] == pytest.approx([1.0, 1.0, 1.0, 0.0])
    assert push._get_hyperedge_size() == 3


@pytest.mark.parametrize('along', [('hyperedges', 1), ('edges', 3), ('hyperedges', 3.0)])
def test_interaction_along_hyperedges_of_a_wrong_size_is_refused(along):
    """Test that hyperedges of fewer than two members, or a size that is not a whole number, are refused."""
    with pytest.raises(ValidationError, match=r"along=\('hyperedges', 3\)"):
        Interaction('push', among='nodes', along=along).prepare()


def test_interaction_along_neighborhoods_adds_up_the_groups_of_a_member():
    """Test that a site adds up a function of every group of a site and its neighbours that holds it."""
    values = np.arange(16.0)
    space = LatticeSpace((4, 4), metric='manhattan', boundary='fixed')
    voters = Element('voters', values={'s': values}, space=space)
    weighted = Interaction(
        'weighted',
        among='voters',
        along='neighborhoods',
        laws=lambda s: [UpdateLaw(s.member.s, s.member.s * total(s.members.s, over=s.members))],
    )
    result = System(elements=[voters], interactions=[weighted]).prepare().simulate(until=1).get_values('voters.s')[1]
    rows, columns = np.divmod(np.arange(16), 4)
    groups = [
        [site, *np.flatnonzero(np.abs(rows - rows[site]) + np.abs(columns - columns[site]) == 1).tolist()]
        for site in range(16)
    ]
    expected = [sum(values[site] * values[group].sum() for group in groups if site in group) for site in range(16)]
    assert np.allclose(result, expected, atol=1e-12)
    assert (weighted._runs_on_edges(), weighted._get_layer()) == (False, None)


class _TriangleGrowthInteraction(Interaction):
    """Growth of one node per step that forms a triangle with two others, and the ageing of every triangle."""

    def __init__(self, label: str, *, among: Role = 'nodes', along: object = ('hyperedges', 3)) -> None:
        super().__init__(label, among=among, along=along)

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the ageing of the triangle."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context.hyperedge.age, context.hyperedge.age + 1)]


def test_interaction_along_hyperedges_of_one_size_runs_on_a_growing_group():
    """Test that triangles added by growth age once per step along the hyperedges of three members."""
    space = NetworkSpace(nx.complete_graph(3), hyperedges=[[0, 1, 2]])
    request = TopologyRequestSpec('add-node', hyperedge=3, independent=True, one=True)
    growth = Interaction('growth', roles={'nodes': 'nodes'}, laws=[UpdateLaw(request, 1.0)])
    system = System(
        elements=[
            Element('nodes', space=space),
            Element('triangles', values={'age': 0.0}, space=space, on='hyperedges'),
        ],
        interactions=[growth, _TriangleGrowthInteraction('ageing')],
    )
    trajectory = system.prepare().simulate(until=3, random_state=0)
    active = trajectory.get_values('triangles._active')[-1] > 0
    assert sorted(trajectory.get_values('triangles.age')[-1][active].tolist()) == [0.0, 1.0, 2.0, 3.0]


def test_interaction_rewires_the_edges_of_a_network():
    """Test that updates on the edges move them away from disagreeing voters at every step until few remain."""
    graph = nx.random_regular_graph(4, 30, seed=1)
    opinions = np.array([0.0, 1.0] * 15)
    voters = Element('voters', values={'opinion': opinions}, space=NetworkSpace(graph))
    system = System(elements=[voters], interactions=[_RewiringInteraction('rewiring')]).prepare()
    trajectory = system.simulate(until=6, random_state=0)
    edges = [
        {tuple(sorted((int(a), int(b)))) for a, b in zip(network.sources, network.targets, strict=True)}
        for network in (trajectory.get_network(frame) for frame in range(len(trajectory.times)))
    ]
    discordant = [sum(opinions[a] != opinions[b] for a, b in frame) for frame in edges]
    assert all(len(frame) == graph.number_of_edges() for frame in edges)
    assert all(edges[i] != edges[i + 1] for i in range(len(edges) - 1) if discordant[i] > 0)
    assert discordant[-1] < discordant[0] / 4


def test_interaction_along_edges_removes_the_edges_whose_ends_disagree():
    """Test that a one-off interaction along edges removes every edge between disagreeing voters in one step."""
    graph = nx.cycle_graph(6)
    opinions = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    voters = Element('voters', values={'opinion': opinions}, space=NetworkSpace(graph))
    pruning = Interaction(
        'pruning',
        among='voters',
        along=True,
        laws=lambda s: [UpdateLaw(TopologyRequestSpec('remove-edge'), (s.source.opinion - s.target.opinion) ** 2)],
    )
    trajectory = System(elements=[voters], interactions=[pruning]).prepare().simulate(until=1, random_state=0)
    network = trajectory.get_network(-1)
    kept = {tuple(sorted((int(a), int(b)))) for a, b in zip(network.sources, network.targets, strict=True)}
    assert kept == {(0, 1), (2, 3), (3, 4), (0, 5)}


class GravityInteraction(Interaction):
    """Gravity on the element of its target, `x'' = -g`."""

    def __init__(self, label: str, *, target: Role, g: float = 9.81) -> None:
        super().__init__(label)
        self.target = target
        self.g = g

    def _prepare(self, context=None):
        """Set the acceleration of the target."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [RateLaw(context.target.x.diff(context.t, 2), -context.g)]


def test_an_interaction_without_a_class_runs_like_a_subclass():
    """Test that a rule given by `roles` and `laws` gives the trajectory of the same rule written as a class."""
    one_off = Interaction('gravity', roles={'target': 'ball'}, laws=lambda s: [RateLaw(s.target.x.diff(s.t, 2), -9.81)])
    trajectories = [
        System(elements=[Element('ball', values={'x': 10.0})], interactions=[interaction])
        .prepare()
        .simulate(until=1.0, dt=0.5)
        for interaction in (one_off, GravityInteraction('gravity', target='ball'))
    ]
    np.testing.assert_allclose(trajectories[0].get_values('ball.x'), trajectories[1].get_values('ball.x'))
    np.testing.assert_allclose(trajectories[0].get_values('ball.x')[-1], 10.0 - 9.81 / 2)


def test_an_interaction_without_a_class_binds_several_roles_and_clones():
    """Test that a predation written without a class binds two roles, and that a clone keeps its roles and laws."""

    def predation(s):
        eaten = 0.1 * s.prey.n * s.predator.n
        return [RateLaw(s.prey.n.diff(s.t), -eaten), RateLaw(s.predator.n.diff(s.t), eaten)]

    interaction = Interaction('predation', roles={'predator': 'foxes', 'prey': 'rabbits'}, laws=predation)
    elements = [Element('foxes', values={'n': 1.0}), Element('rabbits', values={'n': 10.0})]
    system = System(elements=elements, interactions=[clone(interaction)]).prepare()
    assert system.equations_ == {'foxes.n': '0.1*foxes.n*rabbits.n', 'rabbits.n': '-0.1*foxes.n*rabbits.n'}
    assert clone(interaction).roles == {'predator': 'foxes', 'prey': 'rabbits'}


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [({'roles': {'laws': 'ball'}, 'laws': lambda s: []}, 'name of an argument'), ({'laws': 'gravity'}, 'function')],
)
def test_an_interaction_without_a_class_rejects_wrong_arguments(arguments, match):
    """Test that a role named after an argument, or laws that are no function, raise an error."""
    with pytest.raises(ValidationError, match=match):
        System(elements=[Element('ball', values={'x': 1.0})], interactions=[Interaction('rule', **arguments)]).prepare()


def test_an_interaction_without_a_class_binds_among_as_a_role():
    """Test that `among` of a rule written without a class binds its group, which the laws read as `s.among`."""
    cells = Element('cells', values={'n': np.ones(3)}, size=3)
    decay = Interaction('decay', among='cells', laws=lambda s: [RateLaw(s.among.n.diff(s.t), -s.among.n)])
    system = System(elements=[cells], interactions=[decay]).prepare()
    assert system.equations_ == {'cells.n[i]': '-cells.n[i]'}
    assert system.simulate(until=1.0).get_values('cells.n')[-1] == pytest.approx(np.full(3, np.exp(-1.0)), rel=1e-3)


def _push_apart(s):
    """Return equal and opposite velocities of the two members of a pair, along the line between them."""
    separation = s.first.x - s.second.x
    return [RateLaw(s.first.x.diff(s.t), separation), RateLaw(s.second.x.diff(s.t), -separation)]


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_interaction_within_a_radius_runs_once_per_pair_of_two_groups(backend):
    """Test that `within=RadiusNeighborhood(r)` pushes apart only the members of two groups closer than `r`."""
    space = ContinuousSpace(n_dim=1)
    left = Element('left', values={'x': np.array([[0.0], [10.0]])}, size=2, space=space)
    right = Element('right', values={'x': np.array([[1.0]])}, size=1, space=space)
    push = Interaction(
        'push',
        roles={'first': 'left', 'second': 'right'},
        within=RadiusNeighborhood(2.0),
        laws=_push_apart,
    )
    system = System(elements=[left, right], interactions=[push], space=space, backend=backend()).prepare()
    trajectory = system.simulate(until=0.001, dt=0.001)
    assert trajectory.get_values('left.x')[-1].ravel() == pytest.approx([-0.001, 10.0], abs=1e-5)
    assert trajectory.get_values('right.x')[-1].ravel() == pytest.approx([1.001], abs=1e-5)
