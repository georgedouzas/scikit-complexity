"""Test the `_laws` module of `skcomplex.core.laws`."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo, pi, sin
from sage.sets.real_set import RealSet

from skcomplex.core.elements import Element
from skcomplex.core.errors import ConservationError, ValidationError
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import EquationLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from skcomplex.core.laws.utils import derivative, draw, lag, step, where
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.problems import symbol
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import VelocityVerletSolver

from .conftest import PRECISE, MassElement

POSITIVE = RealSet.open(0, oo)


class WrittenElement(Element):
    """Written element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec(order=2)] = 0.3,
        theta_d1: float = 0.0,
        m: Annotated[object, ParameterSpec(domain=POSITIVE)] = 2.0,
        g: Annotated[object, ParameterSpec()] = 9.81,
        l: Annotated[object, ParameterSpec()] = 1.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.theta_d1 = theta_d1
        self.m = m
        self.g = g
        self.l = l

    def _get_laws(self, s):
        """Return the laws of the written element."""
        return [EquationLaw(s.m * s.l * derivative(s.theta, 2) == -s.m * s.g * sin(s.theta))]


class SolvedElement(Element):
    """Solved element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec(order=2)] = 0.3,
        theta_d1: float = 0.0,
        g: Annotated[object, ParameterSpec()] = 9.81,
        l: Annotated[object, ParameterSpec()] = 1.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.theta_d1 = theta_d1
        self.g = g
        self.l = l

    def _get_laws(self, s):
        """Return the laws of the solved element."""
        return [RateLaw(s.theta, -(s.g / s.l) * sin(s.theta))]


class SquaredElement(Element):
    """Squared element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(order=2)] = 1.0,
        x_d1: float = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1

    def _get_laws(self, s):
        """Return the laws of the squared element."""
        return [EquationLaw(derivative(s.x, 2) ** 2, s.x)]


OscillatorsElement = type(
    'OscillatorsElement',
    (Element,),
    {
        **{f'x{index}': state(default=1.0, order=2) for index in range(7)},
        'laws': lambda self, s: [
            EquationLaw(2 * derivative(getattr(s, f'x{index}'), 2), -getattr(s, f'x{index}')) for index in range(7)
        ],
    },
)


class PushInteraction(Interaction):
    """Push interaction."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[object, RoleSpec()],
        f: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.f = f

    def _get_laws(self, s):
        """Return the laws of the push interaction."""
        return [RateLaw(s.target.theta, s.f)]


def _driven_equation(s):
    """Return a mass on a spring pushed by a step at time 1, as an equation."""
    return [EquationLaw(s.m * derivative(s.x, 2) == -s.k * s.x + step(s.t, 1))]


def _driven_rate(s):
    """Return a mass on a spring pushed by a step at time 1, as a rate law."""
    return [RateLaw(s.x, (-s.k * s.x + step(s.t, 1)) / s.m)]


def _switched_equation(s):
    """Return a spring that stiffens on the positive side, as an equation."""
    return [EquationLaw(s.m * derivative(s.x, 2) == where(s.x < 0, -s.k * s.x, -2 * s.k * s.x))]


def _switched_rate(s):
    """Return a spring that stiffens on the positive side, as a rate law."""
    return [RateLaw(s.x, where(s.x < 0, -s.k * s.x, -2 * s.k * s.x) / s.m)]


class EconomyElement(Element):
    """Prices that grow every step, with an adaptive policy rate."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        price: Annotated[float, StateSpec()] = 1.0,
        rate: Annotated[float, ParameterSpec(adaptive=True)] = 0.02,
    ) -> None:
        super().__init__(label, visual=visual)
        self.price = price
        self.rate = rate

    def _get_laws(self, s):
        """Return the growth of prices."""
        return [UpdateLaw(s.price, s.price * 1.03)]


class PolicyInteraction(Interaction):
    """Rule that raises the policy rate whenever prices exceed a target."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return the update of the adaptive rate."""
        return [UpdateLaw(s.target.rate, where(s.target.price > 1.05, s.target.rate + 0.01, s.target.rate))]


class FibonacciElement(Element):
    """Fibonacci element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(lags=1, initial=(1.0,))] = 1.0,
        x_lag1: float = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_lag1 = x_lag1

    def _get_laws(self, s):
        """Return the laws of the fibonacci element."""
        return [UpdateLaw(s.x, s.x + lag(s.x, 1))]


class IncomeElement(Element):
    """Income element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        Y: Annotated[object, StateSpec(lags=1, initial=(100.0,))] = 110.0,
        Y_lag1: float = 100.0,
        c: Annotated[object, ParameterSpec()] = 0.8,
        v: Annotated[object, ParameterSpec()] = 1.5,
        G: Annotated[object, ParameterSpec()] = 20.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.Y = Y
        self.Y_lag1 = Y_lag1
        self.c = c
        self.v = v
        self.G = G

    def _get_laws(self, s):
        """Return the laws of the income element."""
        return [UpdateLaw(s.Y, s.c * (1 + s.v) * s.Y - s.c * s.v * lag(s.Y, 1) + s.G)]


class PumpElement(Element):
    """Pump element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 1.0,
        u: Annotated[object, ParameterSpec(adaptive=True, bounds=(0, 1))] = 0.0,
        weight: Annotated[object, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.u = u
        self.weight = weight

    def _get_laws(self, s):
        """Return the laws of the pump element."""
        return [RateLaw(s.x, -s.x + s.u)]


class PendulumElement(Element):
    """Pendulum element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec(period=2 * pi, order=2)] = 0.1,
        theta_d1: float = 0.0,
        g: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 9.81,
        l: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.theta_d1 = theta_d1
        self.g = g
        self.l = l

    def _get_laws(self, s):
        """Return the laws of the pendulum element."""
        return [RateLaw(s.theta, -(s.g / s.l) * sin(s.theta))]


class SpringInteraction(Interaction):
    """Spring interaction."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[object, RoleSpec()],
        k: Annotated[object, ParameterSpec()] = 8.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.k = k

    def _get_laws(self, s):
        """Return the laws of the spring interaction."""
        return [RateLaw(s.target.x, -s.k * s.target.x / s.target.m)]


class DamperInteraction(Interaction):
    """Damper interaction."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[object, RoleSpec()],
        c: Annotated[object, ParameterSpec()] = 0.4,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.c = c

    def _get_laws(self, s):
        """Return the laws of the damper interaction."""
        return [RateLaw(s.target.x, -s.c * derivative(s.target.x, 1) / s.target.m)]


class StockElement(Element):
    """Stock element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[object, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n

    def _get_laws(self, s):
        """Return the laws of the stock element."""
        return [RateLaw(s.n, -s.n)]


class MismatchedInteraction(Interaction):
    """Mismatched interaction."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[object, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        """Return the laws of the mismatched interaction."""
        return [RateLaw(s.target.x, -s.target.x, order=1)]


def _leaky_laws(s):
    """Return a rate law that changes money outside any flow."""
    return [RateLaw(s.money, -0.1 * s.money)]


class LeakyElement(Element):
    """Leaky element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        money: Annotated[object, StateSpec(quantity='money')] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.money = money

    def _get_laws(self, s):
        """Return the laws of the leaky element."""
        return _leaky_laws(s)


@pytest.mark.parametrize('targets', [(), (symbol('n'), symbol('m'))])
def test_law_of_one_state_rejects_other_targets(targets):
    """Test that a law of one state rejects no target and several targets."""
    with pytest.raises(ValidationError, match='changes one state'):
        RateLaw(targets, 1.0).check()


def test_law_of_one_state_takes_it_alone_or_in_a_tuple():
    """Test that a law of one state takes the state alone or in a one-element tuple, and checks both into a tuple."""
    n = symbol('n')
    assert RateLaw(n, -n).check().targets[0] is n
    assert UpdateLaw((n,), n).check().targets[0] is n
    assert len(UpdateLaw(n, n).check().targets) == 1


def test_noise_rejects_unknown_process():
    """Test that a noise law rejects a process that is neither a label nor a handle."""
    with pytest.raises(ValidationError, match='process'):
        NoiseLaw(symbol('n'), 1.0, process=3).check()


def test_pendulum_as_written_matches_solved_form():
    """Test that `m l θ'' = -m g sin θ` runs exactly like the rate law `θ'' = -(g / l) sin θ`."""
    written = Simulation(System(elements=[WrittenElement('bob')]), solver=PRECISE, dt=0.1).run(until=5).trajectory_
    solved = Simulation(System(elements=[SolvedElement('bob')]), solver=PRECISE, dt=0.1).run(until=5).trajectory_
    assert np.array_equal(written.values('bob.theta'), solved.values('bob.theta'))


def test_rates_add_to_equation():
    """Test that a rate law on a state with an equation adds to the solved highest derivative."""
    pushed = System(
        elements=[WrittenElement('bob', theta=0.0)],
        interactions=[PushInteraction('push', target='bob')],
    )
    trajectory = Simulation(pushed, solver=PRECISE, dt=0.01).run(until=0.1).trajectory_
    assert trajectory.values('bob.theta')[-1] == pytest.approx(0.5 * 0.1**2, rel=1e-2)


def test_nonlinear_highest_derivative_is_rejected():
    """Test that an equation with the square of a highest derivative names the equation and the derivative."""
    with pytest.raises(ValidationError, match=r'equation .* of s is not linear in the highest derivative x_d1'):
        Simulation(System(elements=[SquaredElement('s')])).run(until=1)


def test_many_equations_are_solved_numerically():
    """Test that seven equations, above the symbolic size, are solved per evaluation with the same result."""
    trajectory = Simulation(System(elements=[OscillatorsElement('o')]), solver=PRECISE, dt=0.1).run(until=3).trajectory_
    expected = np.cos(trajectory.times / np.sqrt(2))
    for index in range(7):
        assert np.allclose(trajectory.values(f'o.x{index}'), expected, atol=1e-7)


def test_equation_takes_two_sides():
    """Test that an equation takes either a relation or two sides."""
    assert EquationLaw(1, 2).rhs == 2
    with pytest.raises(ValidationError, match='left and a right side'):
        EquationLaw(1).check()


@pytest.mark.parametrize(
    ('equation', 'rate'),
    [(_driven_equation, _driven_rate), (_switched_equation, _switched_rate)],
    ids=['driver', 'condition'],
)
def test_equation_with_driver_or_condition_matches_rate(equation, rate):
    """Test that equations with a driver or a condition on the right side run like their rate laws."""
    fields = {'states': {'x': state(default=1.0, order=2)}, 'parameters': {'m': 2.0, 'k': 3.0}}
    written = System(
        elements=[type('WrittenSpringElement', (Element,), {**fields, 'laws': lambda self, s: equation(s)})('mass')],
    )
    solved = System(
        elements=[type('SolvedSpringElement', (Element,), {**fields, 'laws': lambda self, s: rate(s)})('mass')],
    )
    first = Simulation(written, solver=PRECISE, dt=0.1).run(until=3).trajectory_.values('mass.x')
    second = Simulation(solved, solver=PRECISE, dt=0.1).run(until=3).trajectory_.values('mass.x')
    assert first == pytest.approx(second, abs=1e-8)


def test_adaptive_parameter_changes_only_when_the_policy_fires():
    """Test that the adaptive rate changes only at steps where prices exceed the target, and starts from its value."""
    system = System(
        elements=[EconomyElement('economy', rate=0.03)],
        interactions=[PolicyInteraction('bank', target='economy')],
    )
    trajectory = Simulation(system).run(until=4).trajectory_
    rates, prices = trajectory.values('economy.rate'), trajectory.values('economy.price')
    assert rates[0] == pytest.approx(0.03)
    np.testing.assert_array_equal(np.diff(rates) > 0, prices[:-1] > 1.05)


def test_fibonacci_recurrence():
    """Test that `x_{t+1} = x_t + x_{t-1}` from 1 and 1 gives the Fibonacci sequence."""
    trajectory = Simulation(System(elements=[FibonacciElement('f')])).run(until=8).trajectory_
    assert trajectory.values('f.x').tolist() == [1, 2, 3, 5, 8, 13, 21, 34, 55]
    assert trajectory.values('f.x_lag1').tolist()[:3] == [1, 1, 2]


def test_multiplier_accelerator_oscillates_with_growing_amplitude():
    """Test that Samuelson's model with c v > 1 oscillates around G / (1 - c) with growing amplitude."""
    income = Simulation(System(elements=[IncomeElement('economy')])).run(until=40).trajectory_.values('economy.Y')
    deviation = income - 20.0 / (1 - 0.8)
    assert np.any(deviation[1:] * deviation[:-1] < 0)
    first, last = np.abs(deviation[:10]).max(), np.abs(deviation[-10:]).max()
    assert last > 10 * first
    roots = np.roots([1, -0.8 * 2.5, 0.8 * 1.5])
    assert np.all(np.abs(roots) > 1)


def test_adaptive_parameters_compile_with_their_bounds():
    """Test that compiling keeps the bounded adaptive parameters, the controls, and leaves the form an ODE."""
    problem = compile_system(System(elements=[PumpElement('pump')]))
    assert problem.form.name == 'ODE'
    assert set(problem.adaptive_parameters) == {'pump.u'}
    assert problem.adaptive_parameters['pump.u'].bounds == (0.0, 1.0)


def test_initial_angular_velocity():
    """Test that a state of order 2 starts from its angle and its angular velocity, both recorded by path."""
    trajectory = (
        Simulation(System(elements=[PendulumElement('bob', theta=0.1, theta_d1=0.5)]), dt=0.1).run(until=1).trajectory_
    )
    assert trajectory.values('bob.theta')[0] == pytest.approx(0.1)
    assert trajectory.values('bob.theta_d1')[0] == pytest.approx(0.5)
    assert trajectory.values('bob.theta')[1] > 0.1


def test_first_order_unchanged():
    """Test that a state without an order compiles without companion states."""
    simulation = Simulation(System(elements=[StockElement('a')]), dt=0.5).run(until=1)
    assert simulation.problem_.states.names == ['a.n']


def test_newton_superposition_of_separate_laws():
    """Test that a spring and a damper from separate interactions add up to `m x'' = -k x - c x'`."""
    system = System(
        elements=[MassElement('mass')],
        interactions=[SpringInteraction('spring', target='mass'), DamperInteraction('damper', target='mass')],
    )
    trajectory = Simulation(system, solver=PRECISE, dt=0.1).run(until=10).trajectory_
    gamma = 0.4 / (2 * 2.0)
    omega = np.sqrt(8.0 / 2.0 - gamma**2)
    t = trajectory.times
    expected = np.exp(-gamma * t) * (np.cos(omega * t) + gamma / omega * np.sin(omega * t))
    assert np.allclose(trajectory.values('mass.x'), expected, atol=1e-7)


def test_order_mismatch_is_rejected():
    """Test that a law asserting order 1 on a state of order 2 names the law, the state, and both orders."""
    system = System(elements=[MassElement('mass')], interactions=[MismatchedInteraction('wrong', target='mass')])
    with pytest.raises(
        ValidationError,
        match=r'wrong on mass\.x gives the derivative of order 1, but the state has order 2',
    ):
        Simulation(system).run(until=1)


def test_velocity_verlet_runs_order_two_position():
    """Test that velocity Verlet recognises a position of order 2 and conserves the oscillator's energy."""
    system = System(elements=[MassElement('mass')], interactions=[SpringInteraction('spring', target='mass')])
    trajectory = Simulation(system, solver=VelocityVerletSolver(dt=0.001), dt=0.1).run(until=10).trajectory_
    energy = 0.5 * 2.0 * trajectory.values('mass.x_d1') ** 2 + 0.5 * 8.0 * trajectory.values('mass.x') ** 2
    assert np.allclose(energy, energy[0], rtol=1e-4)


def test_undeclared_change_names_the_state_and_the_quantity():
    """Test that a rate law on a state tagged money raises a conservation error naming both."""
    with pytest.raises(ConservationError, match=r"leaky\.money.*'money'"):
        compile_system(System(elements=[LeakyElement('leaky')]))


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'waiting': draw('wait', 'expon'), 'changes': {symbol('n'): 1}}, 'changes belong to the exponential clock'),
        ({'rate': 1.0, 'changes': {symbol('n'): 1}, 'reset': {symbol('n'): 0}}, 'resets and labels to the waiting'),
        ({'rate': 1.0, 'changes': {symbol('n'): 1}, 'label': 'fires'}, 'takes a label'),
    ],
)
def test_jump_rejects_option_of_other_clock(arguments, message):
    """Test that a jump rejects at construction an option of the clock it does not use."""
    with pytest.raises(ValidationError, match=message):
        JumpLaw(**arguments).check()
