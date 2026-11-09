"""Test the symbolic analysis of systems."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR, cos, diff, function, sin, sqrt

from skcomplex.core.analysis import symbolic
from skcomplex.core.components.elements import Element, NonnegativeElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import derivative, lag
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.compile.passes._operator_compiler import OperatorCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import AnalysisError
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from skcomplex.systems.biology.common.populations import LogisticGrowthInteraction, PopulationElement
from skcomplex.systems.biology.ecology import LogisticGrowthSystem
from skcomplex.systems.biology.epidemiology import RecoveryInteraction
from skcomplex.systems.economics.finance import AssetElement, GeometricBrownianMotionInteraction
from skcomplex.systems.economics.markets import CobwebInteraction, MarketElement
from skcomplex.systems.mathematics.fractals import KochCurveSystem
from skcomplex.systems.physics.active_matter import CuckerSmaleFlockingSystem
from skcomplex.systems.physics.common.particles import (
    ConstantForceInteraction,
    HarmonicPotentialInteraction,
    MassPointElement,
)
from skcomplex.systems.physics.statistical_mechanics import QuantumHeisenbergSystem

from ...conftest import CallableGrowthInteraction, HoldingElement


def test_equations_of_logistic_growth():
    """Test that logistic growth gives `dn/dt = r * n * (1 - n / K)`."""
    system = System(
        elements=[PopulationElement('p', x=10)],
        interactions=[LogisticGrowthInteraction('g', target='p')],
    )
    problem = PassCompiler().prepare(system).run().problem_
    n = function('x_p')(problem.time)
    r, K = problem.symbols['g.r'], problem.symbols['g.K']
    equation = symbolic(system).equations['p.x']
    assert equation.is_relational()
    assert bool(equation.lhs() == n.diff(problem.time))
    assert (equation.rhs() - r * n * (1 - n / K)).simplify_full() == 0


def test_equations_of_discrete_map_gives_next_value():
    """Test that a discrete map gives the value at `t + 1`."""
    system = System(elements=[MarketElement('m')], interactions=[CobwebInteraction('c', target='m')])
    equation = symbolic(system).equations['m.price']
    t = PassCompiler().prepare(system).run().problem_.time
    assert bool(equation.lhs() == function('price_m')(t + 1))


def test_equations_of_group_with_numeric_derivatives_raises_error():
    """A group of order two whose forces are computed numerically has no symbolic equations."""
    with pytest.raises(AnalysisError, match='computed numerically'):
        symbolic(CuckerSmaleFlockingSystem().prepare().system_)


def test_equations_of_jump_process_raises_error():
    """Test that jump processes have no symbolic equations."""
    system = System(
        elements=[NonnegativeElement('I', x=1), NonnegativeElement('R')],
        interactions=[RecoveryInteraction('r', infected='I', recovered='R', stochastic=True)],
    )
    with pytest.raises(AnalysisError, match='jump process'):
        symbolic(system)


def test_solve_symbolically_free_particle():
    """Test that a free particle moves linearly with two integration constants."""
    system = System(elements=[MassPointElement('p')], space=ContinuousSpace(n_dim=1))
    solution = symbolic(system, kind='solve').solutions['p.x[0]']
    t = PassCompiler().prepare(system).run().problem_.time
    slope, start = solution.rhs().diff(t), solution.rhs().subs({t: 0})
    assert solution.rhs().diff(t, 2) == 0
    assert not slope.is_numeric()
    assert not start.is_numeric()
    assert (solution.rhs() - (slope * t + start)).simplify_full() == 0


def test_solve_symbolically_constant_acceleration():
    """Test that a constant force gives `F * t**2 / (2 * m)` from rest."""
    system = System(
        elements=[MassPointElement('p')],
        interactions=[ConstantForceInteraction('push', target='p', F=3.0)],
        space=ContinuousSpace(n_dim=1),
    )
    problem = PassCompiler().prepare(system).run().problem_
    (F,) = problem.symbols['push.F']
    m = problem.symbols['p.m']
    solution = symbolic(system, kind='solve', use_initial_state=True).solutions['p.x[0]']
    assert (solution.rhs() - F * problem.time**2 / (2 * m)).simplify_full() == 0


def test_solve_symbolically_harmonic_oscillator():
    """Test that an oscillator from its initial state oscillates with frequency `sqrt(k / m)`."""
    system = System(
        elements=[MassPointElement('p', x=1.0, x_d1=0.0)],
        interactions=[HarmonicPotentialInteraction('trap', target='p')],
        space=ContinuousSpace(n_dim=1),
    )
    problem = PassCompiler().prepare(system).run().problem_
    k, m, t = problem.symbols['trap.k'], problem.symbols['p.m'], problem.time
    (center,) = problem.symbols['trap.center']
    solution = symbolic(system, kind='solve', use_initial_state=True).solutions['p.x[0]'].rhs().subs({center: 0})
    expected = cos(sqrt(k / m) * t)
    values = {k: 2.0, m: 0.5}
    for time in np.linspace(0, 5, 6):
        assert float(solution.subs(values).subs({t: time})) == pytest.approx(
            float(expected.subs(values).subs({t: time})),
        )


def test_solve_symbolically_not_ode_raises_error():
    """Test that only ODE systems are solved."""
    system = System(
        elements=[AssetElement('s')],
        interactions=[GeometricBrownianMotionInteraction('g', target='s')],
    )
    with pytest.raises(AnalysisError, match='stochastic differential equations'):
        symbolic(system, kind='solve')


def test_solve_symbolically_callable_raises_error():
    """Test that callable laws cannot be solved."""
    system = System(
        elements=[HoldingElement('rabbits', n=1.0)],
        interactions=[CallableGrowthInteraction('g', target='rabbits')],
    )
    with pytest.raises(AnalysisError, match='callable'):
        symbolic(system, kind='solve')


def test_sage_expression_compiles_like_number():
    """Test that a constant force given as a Sage expression simulates like the same force given as a number."""
    force = SR(2) * SR(1.5)
    system = System(
        elements=[MassPointElement('p')],
        interactions=[ConstantForceInteraction('push', target='p', F=float(force))],
        space=ContinuousSpace(n_dim=1),
    )

    class SageForceInteraction(Interaction):
        """Force written as a Sage expression."""

        def __init__(
            self,
            label: str,
            *,
            target: Annotated[str, RoleSpec()],
        ) -> None:
            super().__init__(label)
            self.target = target

        def _get_laws(self, s):
            """Return a rate with a Sage value."""
            return [RateLaw(s.target.x, [force])]

    sage_system = System(
        elements=[MassPointElement('p')],
        interactions=[SageForceInteraction('push', target='p')],
        space=ContinuousSpace(n_dim=1),
    )
    expected = simulate(system, until=1).values('p.x')
    result = simulate(sage_system, until=1).values('p.x')
    assert np.allclose(expected, result)


class PendulumElement(Element):
    """Pendulum element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec(order=2)] = 0.1,
        theta_d1: float = 0.0,
        g: Annotated[object, ParameterSpec()] = 9.81,
        l: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.theta_d1 = theta_d1
        self.g = g
        self.l = l

    def _get_laws(self, s):
        """Return the laws of the pendulum element."""
        return [RateLaw(s.theta, -(s.g / s.l) * sin(s.theta))]


class SmallAngleElement(Element):
    """Small angle element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec(order=2)] = 0.1,
        theta_d1: float = 0.0,
        g: Annotated[object, ParameterSpec()] = 9.81,
        l: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.theta_d1 = theta_d1
        self.g = g
        self.l = l

    def _get_laws(self, s):
        """Return the laws of the small angle element."""
        return [RateLaw(s.theta, -(s.g / s.l) * s.theta)]


class ThirdOrderElement(Element):
    """Third order element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(order=3, initial=(1.0, 0.0))] = 0.0,
        x_d2: float = 0.0,
        x_d1: float = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d2 = x_d2
        self.x_d1 = x_d1

    def _get_laws(self, s):
        """Return the laws of the third order element."""
        return [RateLaw(s.x, -derivative(s.x, 1))]


class FirstOrderPendulumElement(Element):
    """First order pendulum element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec()] = 0.1,
        omega: Annotated[object, StateSpec()] = 0.0,
        g: Annotated[object, ParameterSpec()] = 9.81,
        l: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.omega = omega
        self.g = g
        self.l = l

    def _get_laws(self, s):
        """Return the laws of the first order pendulum element."""
        return [RateLaw(s.theta, s.omega), RateLaw(s.omega, -(s.g / s.l) * sin(s.theta))]


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


def test_pendulum_equation_keeps_its_order():
    """Test that a pendulum of order 2 gives one second-order equation and no separate velocity equation."""
    equations = symbolic(System(elements=[PendulumElement('bob')])).equations
    assert list(equations) == ['bob.theta']
    theta = function('theta_bob')(SR.var('t'))
    g, l = SR.var('g_bob'), SR.var('l_bob')
    equation = equations['bob.theta']
    assert bool(equation.lhs() == diff(theta, SR.var('t'), 2))
    assert (equation.rhs() + g * sin(theta) / l).simplify_full() == 0


def test_lagged_equation_reads_past_values():
    """Test that a difference equation with a lag reads the value one step back."""
    equation = symbolic(System(elements=[FibonacciElement('f')])).equations['f.x']
    t = SR.var('t')
    x = function('x_f')
    assert bool(equation.lhs() == x(t + 1))
    assert (equation.rhs() - x(t) - x(t - 1)).simplify_full() == 0


def test_small_angle_pendulum_solved_with_plain_numbers():
    """Test that `θ'' = -(g / l) θ` with plain positive numbers solves to `0.1 cos(sqrt(g / l) t)`."""
    solution = symbolic(System(elements=[SmallAngleElement('bob')]), kind='solve', use_initial_state=True).solutions[
        'bob.theta'
    ]
    values = {SR.var('g_bob'): 9.81, SR.var('l_bob'): 2.0}
    for time in (0.3, 1.0, 2.5):
        expected = 0.1 * float(cos(sqrt(9.81 / 2.0) * time))
        assert float(solution.rhs().subs(values).subs({SR.var('t'): time})) == pytest.approx(expected)


def test_third_order_equation_solved():
    """Test that `x''' = -x'` from `x = 0`, `x' = 1`, `x'' = 0` solves to `sin(t)`."""
    solution = symbolic(System(elements=[ThirdOrderElement('a')]), kind='solve', use_initial_state=True).solutions[
        'a.x'
    ]
    t = SR.var('t')
    for time in (0.3, 1.0, 2.5):
        assert float(solution.rhs().subs({t: time})) == pytest.approx(float(sin(time)))


@pytest.mark.parametrize(
    'element',
    [PendulumElement('bob'), FirstOrderPendulumElement('bob')],
    ids=['second-order', 'first-order'],
)
def test_nonlinear_pendulum_has_no_false_solution(element):
    """Test that the nonlinear pendulum raises the no-closed-form error instead of returning unevaluated transforms."""
    with pytest.raises(AnalysisError, match='No closed form was found'):
        symbolic(System(elements=[element]), kind='solve', use_initial_state=True)


def test_remaining_sign_question_names_parameter():
    """Test that a sign Maxima still needs raises an error that names the parameter and the domain to declare."""

    class SpringElement(Element):
        """Spring element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec(order=2)] = 0.1,
            x_d1: float = 0.0,
            k: Annotated[object, ParameterSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x
            self.x_d1 = x_d1
            self.k = k

        def _get_laws(self, s):
            """Return the laws of the spring element."""
            return [RateLaw(s.x, -s.k * s.x)]

    with pytest.raises(AnalysisError, match=r'sign of k_a.*domain=RealSet'):
        symbolic(System(elements=[SpringElement('a')]), kind='solve', use_initial_state=True)


def test_symbolic_refuses_rewriting_systems_with_an_analysis_error():
    """Test that a system of words raises the analysis error of a form without equations, not a compiling error."""
    with pytest.raises(AnalysisError, match='rewrites words or patches'):
        symbolic(KochCurveSystem())


def test_hamiltonian_prints_operators_and_equations_refuse():
    """Test that an operator system prints its Hamiltonian and has no state equations."""
    system = QuantumHeisenbergSystem(n_sites=2, boundary='open').prepare().system_
    problem = OperatorCompiler().prepare(system).run().problem_
    hamiltonian = symbolic(problem, kind='hamiltonian').hamiltonian
    assert {'J_exchange', 'delta_exchange', 'h_field'} <= {str(variable) for variable in hamiltonian.variables()}
    assert 'local_operator(psi_chain_0, x)*local_operator(psi_chain_1, x)' in str(hamiltonian)
    with pytest.raises(AnalysisError, match='hamiltonian'):
        symbolic(problem)


def test_settings_the_kind_ignores_are_rejected():
    """Test that the initial state, which only solving reads, is rejected for the equations."""
    with pytest.raises(AnalysisError, match='does not read use_initial_state'):
        symbolic(LogisticGrowthSystem().prepare().system_, use_initial_state=True)
