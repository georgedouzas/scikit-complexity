"""Test the `_interaction` module of `skcomplex.core.interactions`."""

from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet
from sklearn.base import clone

from skcomplex.core.elements import BaseElement, Element, GroupElement, LatticeElement
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import NoiseLaw, RateLaw
from skcomplex.core.laws.utils import neighbors
from skcomplex.core.spaces import NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import EulerMaruyamaSolver
from skcomplex.errors import LabelError, RoleError
from skcomplex.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec, WienerProcessSpec

from ...conftest import HoldingElement, _correlated


class StockElement(Element):
    """Stock with one nonnegative state."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
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
    return System(
        elements=[HoldingElement('a'), HoldingElement('b')],
        interactions=[
            ShockInteraction('first', target='a', noise=first),
            ShockInteraction('second', target='b', noise=second),
        ],
    )


def test_interaction_has_no_laws_by_default():
    """Test that an interaction that returns no laws of its own runs into none."""
    assert Interaction('empty').check().prepare().run(SimpleNamespace()).laws_ == []


def test_interaction_requires_its_roles():
    """Test that an interaction without a required role cannot be made."""
    with pytest.raises(TypeError, match='predator'):
        PredationInteraction('hunt', prey='rabbits')


def test_interaction_clones_and_pickles_with_its_values():
    """Test that an interaction clones and pickles with its roles and parameters."""
    growth = GrowthInteraction('g', target='a', r=0.7)
    assert clone(growth).get_params() == {'label': 'g', 'r': 0.7, 'target': 'a'}
    data = ForkingPickler.dumps(System(elements=[StockElement('a', n=3.0)], interactions=[growth]))
    restored = ForkingPickler.loads(data)
    assert restored.elements[0].n == 3.0
    assert restored.interactions[0].r == 0.7


def test_role_binds_to_group_and_applies_to_every_member():
    """Test that an interaction bound to a group applies its law to every member."""
    system = System(
        elements=[GroupElement('herds', element=StockElement, size=3, values={'n': [1.0, 2.0, 3.0]})],
        interactions=[GrowthInteraction('growth', target='herds', r=2.0)],
    )
    problem = compile_system(system)
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert drift.tolist() == [2.0, 4.0, 6.0]


def test_group_of_members_grows_like_one_member():
    """Test that a group of 100 members grows every member at the rate of the interaction bound to it."""
    system = System(
        elements=[GroupElement('herd', element=StockElement, size=100, values={'n': np.linspace(1, 2, 100)})],
        interactions=[GrowthInteraction('g', target='herd')],
    )
    trajectory = Simulation(system, dt=1).check().prepare().run(until=1).trajectory_
    assert trajectory.values('herd.n').shape == (2, 100)
    assert np.allclose(trajectory.values('herd.n')[-1], np.linspace(1, 2, 100) * np.exp(0.5), rtol=1e-3)


def test_single_element_bound_with_group_receives_sum():
    """Test that a single element bound next to a group receives the sum over the members."""
    system = System(
        elements=[
            GroupElement('rabbits', element=StockElement, size=3, values={'n': [1.0, 2.0, 3.0]}),
            StockElement('fox', n=2.0),
        ],
        interactions=[PredationInteraction('hunt', predator='fox', prey='rabbits')],
    )
    problem = compile_system(system)
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert drift.tolist() == [-2.0, -4.0, -6.0, 12.0]
    assert problem.has_group_sums


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
def test_interaction_reads_two_groups_member_by_member(backend):
    """Test that an interaction bound to two groups of equal size reads row `i` of both together."""
    system = System(
        elements=[
            GroupElement('source', element=StockElement, size=5, values={'n': np.arange(5.0)}),
            GroupElement('sink', element=StockElement, size=5),
        ],
        interactions=[TransferInteraction('flow', source='source', sink='sink')],
    )
    problem = compile_system(system, backend=backend)
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert problem.states.read(drift, 'sink.n').tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_interaction_reading_groups_of_different_sizes_raises_error():
    """Test that an interaction reading groups of different sizes member by member names both groups and sizes."""
    system = System(
        elements=[
            GroupElement('source', element=StockElement, size=5),
            GroupElement('sink', element=StockElement, size=7),
        ],
        interactions=[TransferInteraction('flow', source='source', sink='sink')],
    )
    with pytest.raises(RoleError, match=r'sink of 7 members, source of 5 members'):
        compile_system(system)


def test_correlated_noise():
    """Test that two noise laws drawing the processes of a set bound to a role have the increments `h C`."""
    covariance = [[1.0, 0.6], [0.6, 2.0]]
    simulation = (
        Simulation(_correlated(covariance), solver=EulerMaruyamaSolver(dt=0.5), random_state=0, dt=0.5)
        .check()
        .prepare()
    )
    trajectory = simulation.run(until=5_000).trajectory_
    increments = np.diff(np.stack([trajectory.values('a.n'), trajectory.values('b.n')]), axis=1)
    np.testing.assert_allclose(np.cov(increments), 0.5 * np.asarray(covariance), atol=0.05)
    assert not issubclass(WienerProcessSpec, BaseElement)


def test_interactions_share_processes_of_one_label():
    """Test that two interactions that bind processes of one label draw the same increments."""
    market = WienerProcessSpec('market')
    trajectory = (
        Simulation(_shocked(market, WienerProcessSpec('market')), random_state=0)
        .check()
        .prepare()
        .run(until=1)
        .trajectory_
    )
    np.testing.assert_array_equal(trajectory.values('a.n'), trajectory.values('b.n'))
    assert compile_system(_shocked(market, market)).n_processes == 1


def test_processes_of_one_label_must_agree():
    """Test that two different sets of processes with one label, and a path bound to a noise role, are rejected."""
    with pytest.raises(LabelError, match='two different Wiener processes'):
        compile_system(_shocked(WienerProcessSpec('market'), WienerProcessSpec('market', dimension=2)))
    with pytest.raises(LabelError, match='label of another child'):
        compile_system(_shocked(WienerProcessSpec('a'), WienerProcessSpec('a')))
    with pytest.raises(RoleError, match='takes a WienerProcessSpec'):
        compile_system(_shocked('market', 'market'))


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
def test_edge_states_weigh_neighbourhood_reductions(backend):
    """Test that a reduction over a network weighs every neighbour with the state of the edge that joins them."""
    space = NetworkSpace(nx.path_graph(4))
    system = System(
        elements=[
            LatticeElement('nodes', element=StockElement, space=space, values={'n': [1.0, 2.0, 3.0, 4.0]}),
            GroupElement('links', element=StockElement, space=space, on='edges', values={'n': [10.0, 100.0, 1000.0]}),
        ],
        interactions=[WeightedInteraction('weighted', nodes='nodes', links='links')],
    )
    problem = compile_system(system, backend=backend)
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    expected = [10.0 * 2, 10.0 * 1 + 100.0 * 3, 100.0 * 2 + 1000.0 * 4, 1000.0 * 3]
    assert np.allclose(problem.states.read(drift, 'nodes.n'), expected, atol=1e-9)
