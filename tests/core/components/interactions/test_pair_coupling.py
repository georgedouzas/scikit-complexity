"""Test the `_pair_coupling` module of `skcomplex.core.components.interactions`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
import pytest
from sage.all__sagemath_symbolics import exp, oo, sin, sqrt, var, vector
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element, GroupElement, PointElement
from skcomplex.core.components.interactions import PairCouplingInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import compile_system
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec


class MassPointElement(PointElement):
    """Point with a mass."""

    def __init__(
        self,
        label: str,
        x: npt.ArrayLike = 0.0,
        x_d1: npt.ArrayLike = 0.0,
        visual: VisualSpec | None = None,
        *,
        m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, x=x, x_d1=x_d1, visual=visual)
        self.m = m


class WalkerElement(Element):
    """Member with a vector state of order 1 and no dynamics of its own."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        y: Annotated[object, StateSpec(vector=True, role='position')] = 0.0,
        phase: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.y = y
        self.phase = phase


class BondPairInteraction(PairCouplingInteraction):
    """Pair coupling that binds its members as `left` and `right`."""

    def __init__(
        self,
        label: str,
        left: Annotated[str, RoleSpec()],
        right: Annotated[str, RoleSpec()],
        cutoff: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        **arguments: object,
    ) -> None:
        super().__init__(label, left, right, cutoff=cutoff, **arguments)
        self.left = left
        self.right = right

    def _get_role(self, role: str) -> str:
        """Return `left` for the first member and `right` for the second."""
        return {'first': 'left', 'second': 'right'}.get(role, role)


def _morse(namespace, r):
    """Return the Morse energy of depth 2, width 1.5, and minimum at 1."""
    return 2 * (1 - exp(-1.5 * (r - 1))) ** 2


def _morse_energy(first, second):
    """Return the numeric Morse energy of two members at their positions."""
    r = np.linalg.norm(first - second)
    return 2 * (1 - np.exp(-1.5 * (r - 1))) ** 2


def _soft(namespace, r):
    """Return the soft repulsion energy `(1 - r) ** 2 / 2` of a unit cutoff."""
    return (1 - r) ** 2 / 2


def _rates(system, path):
    """Return the rates of a state at the initial state, one row per member."""
    problem = compile_system(system)
    entry = problem.states.entry(path)
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    return drift[entry.start : entry.start + entry.size].reshape(entry.rows, entry.columns)


def _pair_accelerations(positions, masses, cutoff=3.0):
    """Return the compiled accelerations of a pair of points under the Morse potential."""
    group = GroupElement('pair', element=MassPointElement, size=2, values={'x': positions, 'm': masses})
    system = System(
        elements=[group],
        interactions=[PairCouplingInteraction('morse', 'pair', 'pair', cutoff=cutoff, potential=_morse, divisor='m')],
        space=ContinuousSpace(n_dim=2),
    )
    return _rates(system, 'pair.x_d1')


def test_pair_coupling_potential_gives_minus_the_gradient_over_the_divisor():
    """Test that the second derivative of each member is minus the gradient of the potential, divided by its mass."""
    positions, masses = np.array([[0.0, 0.0], [1.2, 0.9]]), np.array([1.0, 2.0])
    step = 1e-6
    gradient = np.array(
        [
            (_morse_energy(positions[0] + shift, positions[1]) - _morse_energy(positions[0] - shift, positions[1]))
            / (2 * step)
            for shift in step * np.eye(2)
        ],
    )
    accelerations = _pair_accelerations(positions, masses)
    np.testing.assert_allclose(accelerations[0], -gradient / masses[0], rtol=1e-6)
    np.testing.assert_allclose(accelerations[1], gradient / masses[1], rtol=1e-6)


def test_pair_coupling_vanishes_beyond_the_cutoff():
    """Test that members farther apart than the cutoff do not couple."""
    accelerations = _pair_accelerations(np.array([[0.0, 0.0], [2.0, 0.0]]), np.array([1.0, 1.0]), cutoff=1.5)
    np.testing.assert_allclose(accelerations, 0.0)


def test_pair_coupling_holds_members_at_the_potential_minimum():
    """Test that two members at the minimum of the potential stay at rest."""
    group = GroupElement('pair', element=PointElement, size=2, values={'x': [[0.0, 0.0], [1.0, 0.0]]})
    system = System(
        elements=[group],
        interactions=[PairCouplingInteraction('morse', 'pair', 'pair', cutoff=3.0, potential=_morse)],
        space=ContinuousSpace(n_dim=2),
    )
    trajectory = Simulation(system, solver=ScipyODESolver(), dt=0.1).prepare().run(until=1).trajectory_
    np.testing.assert_allclose(trajectory.values('pair.x_d1'), 0.0, atol=1e-12)


@pytest.mark.slow
def test_pair_coupling_conserves_the_total_first_derivative():
    """Test that soft repulsion inside an isolated group keeps the sum of the first derivatives of the positions."""
    rng = np.random.default_rng(0)
    values = {'x': rng.uniform(0, 3, (200, 2)), 'x_d1': rng.normal(0, 1, (200, 2))}
    system = System(
        elements=[GroupElement('gas', element=PointElement, size=200, values=values)],
        interactions=[PairCouplingInteraction('soft', 'gas', 'gas', cutoff=0.2, potential=_soft)],
        space=ContinuousSpace(n_dim=2),
    )
    solver = ScipyODESolver(rtol=1e-9, atol=1e-12)
    trajectory = Simulation(system, solver=solver, dt=0.05).prepare().run(until=2).trajectory_
    momentum = trajectory.values('gas.x_d1').sum(axis=1)
    np.testing.assert_allclose(momentum, np.broadcast_to(momentum[0], momentum.shape), atol=1e-8)


def test_pair_coupling_potential_moves_a_state_of_order_one():
    """Test that a potential on a state of order 1 gives its rate, minus the gradient, on the named state."""
    group = GroupElement('walkers', element=WalkerElement, size=2, values={'y': [[0.0, 0.0], [0.5, 0.0]]})
    coupling = PairCouplingInteraction('soft', 'walkers', 'walkers', position='y', write='y', potential=_soft)
    rates = _rates(System(elements=[group], interactions=[coupling], space=ContinuousSpace(n_dim=2)), 'walkers.y')
    np.testing.assert_allclose(rates, [[-0.5, 0.0], [0.5, 0.0]])


def test_pair_coupling_function_adds_its_value_to_both_members_with_the_roles_swapped():
    """Test that a function of the pair gives each member its own value, with the other member as `second`."""
    group = GroupElement(
        'walkers',
        element=WalkerElement,
        size=2,
        values={'y': [[0.0, 0.0], [0.5, 0.0]], 'phase': [0.0, 1.0]},
    )
    coupling = PairCouplingInteraction(
        'sync',
        'walkers',
        'walkers',
        position='y',
        write='phase',
        function=lambda s: sin(s.second.phase - s.first.phase),
    )
    system = System(elements=[group], interactions=[coupling], space=ContinuousSpace(n_dim=2))
    np.testing.assert_allclose(_rates(system, 'walkers.phase').ravel(), [np.sin(1.0), -np.sin(1.0)])


def test_pair_coupling_laws_are_symbolic_and_opposite_for_a_potential():
    """Test that the laws of a potential give opposite values to the two members with equal divisors."""
    x1, x2 = var('x1 x2')
    first, second = SimpleNamespace(x=vector([x1]), m=1), SimpleNamespace(x=vector([x2]), m=1)
    coupling = PairCouplingInteraction('soft', 'gas', 'gas', potential=_soft, divisor='m').prepare()
    push, pull = coupling.run(SimpleNamespace(first=first, second=second)).laws_
    assert (push.target, pull.target) == (first.x, second.x)
    assert bool((push.value + pull.value)[0] == 0)
    assert bool(push.value[0].subs({x1: 0.0, x2: 0.5}) == -(1 - sqrt(0.25)) * 1.0)


@pytest.mark.parametrize(
    'arguments',
    [{}, {'function': lambda s: 0, 'potential': _soft}, {'function': 1.0}, {'potential': 1.0}],
)
def test_check_rejects_not_exactly_one_callable(arguments):
    """Test that `check` rejects a coupling without exactly one callable of the function and the potential."""
    with pytest.raises(ValidationError, match='exactly one'):
        PairCouplingInteraction('pair', 'gas', 'gas', **arguments).prepare()


def test_pair_coupling_function_writes_several_states_jointly():
    """Test that a function of several states adds one value per state to both members, with the roles swapped."""
    x1, x2, a, b = var('x1 x2 a b')
    first, second = SimpleNamespace(x=vector([x1]), theta=a), SimpleNamespace(x=vector([x2]), theta=b)
    coupling = PairCouplingInteraction(
        'joint',
        'units',
        'units',
        write=('x', 'theta'),
        function=lambda s: (s.second.x - s.first.x, sin(s.second.theta - s.first.theta)),
    )
    laws = coupling.prepare().run(SimpleNamespace(first=first, second=second)).laws_
    assert [law.target for law in laws] == [first.x, first.theta, second.x, second.theta]
    assert [law.value for law in laws] == [vector([x2 - x1]), sin(b - a), vector([x1 - x2]), sin(a - b)]


def test_check_rejects_a_potential_of_several_states():
    """Test that `check` rejects a potential with several states to write."""
    with pytest.raises(ValidationError, match='one state'):
        PairCouplingInteraction('pair', 'gas', 'gas', write=('x', 'v'), potential=_soft).prepare()


def test_get_laws_reads_renamed_pair_members():
    """Test that a pair coupling that binds `left` and `right` gives the rates of the plain pair coupling."""
    group = GroupElement(
        'pair',
        element=MassPointElement,
        size=2,
        values={'x': [[0.0, 0.0], [1.2, 0.9]], 'm': [1.0, 2.0]},
    )
    walkers = GroupElement(
        'walkers',
        element=WalkerElement,
        size=2,
        values={'y': [[0.0, 0.0], [0.5, 0.0]], 'phase': [0.0, 1.0]},
    )
    rates = []
    for interactions in (
        [
            PairCouplingInteraction('morse', 'pair', 'pair', cutoff=3.0, potential=_morse, divisor='m'),
            PairCouplingInteraction(
                'sync',
                'walkers',
                'walkers',
                position='y',
                write='phase',
                function=lambda s: sin(s.second.phase - s.first.phase),
            ),
        ],
        [
            BondPairInteraction('morse', 'pair', 'pair', cutoff=3.0, potential=_morse, divisor='m'),
            BondPairInteraction(
                'sync',
                'walkers',
                'walkers',
                position='y',
                write='phase',
                function=lambda s: sin(s.right.phase - s.left.phase),
            ),
        ],
    ):
        system = System(elements=[group, walkers], interactions=interactions, space=ContinuousSpace(n_dim=2))
        rates.append(np.concatenate([_rates(system, 'pair.x_d1').ravel(), _rates(system, 'walkers.phase').ravel()]))
    np.testing.assert_allclose(rates[1], rates[0])
    assert np.abs(rates[0]).sum() > 0
