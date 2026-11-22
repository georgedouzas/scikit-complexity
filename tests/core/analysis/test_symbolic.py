"""Test the symbolic analysis of systems."""

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR, cos, diff, function, oo, sin, sqrt
from sage.sets.real_set import RealSet

from skcomplex.core.analysis import symbolic
from skcomplex.core.analysis._symbolic import _MittagLefflerFunction
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import fractional
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.errors import AnalysisError
from skcomplex.core.systems import System
from skcomplex.core.types import Role
from skcomplex.systems.biology.common.populations import LogisticGrowthInteraction, PopulationElement
from skcomplex.systems.biology.ecology import LogisticGrowthSystem
from skcomplex.systems.biology.epidemiology import RecoveryInteraction
from skcomplex.systems.economics.finance import AssetElement, GeometricBrownianMotionInteraction
from skcomplex.systems.economics.markets import CobwebInteraction, MarketElement
from skcomplex.systems.mathematics.fractals import KochCurveSystem
from skcomplex.systems.physics.common.particles import (
    ConstantForceInteraction,
)
from skcomplex.systems.physics.common.quantum import QuantumCompiler
from skcomplex.systems.physics.statistical_mechanics import LennardJonesFluidSystem, QuantumHeisenbergSystem
from skcomplex.systems.physics.waves import HarmonicPotentialInteraction

from ...conftest import BodyElement, CallableGrowthInteraction, HoldingElement


class AmountElement(Element):
    """Element that holds one nonnegative scalar `x`, an amount."""

    def __init__(self, label: str, *, x: Annotated[float, RealSet.closed_open(0, oo)] = 0.0) -> None:
        super().__init__(label)
        self.x = x


def _system(element, rates):
    """Return a system of one element whose laws are a function of the namespace of its role `target`."""
    return System(elements=[element], interactions=[Interaction('laws', roles={'target': element.label}, laws=rates)])


def _relaxing():
    """Return a value that relaxes with a Caputo derivative of order 0.6."""
    element = Element('r', values={'x': 2.0, 'lam': 0.5})
    return _system(element, lambda s: [RateLaw(fractional(s.target.x, 0.6), -s.target.lam * s.target.x)])


def _saturating():
    """Return a fractional value that follows a nonlinear law."""
    element = Element('s', values={'x': 0.5})
    return _system(element, lambda s: [RateLaw(fractional(s.target.x, 0.6), s.target.x * (1 - s.target.x))])


def _free_particle(*interactions):
    """Return a body at a position of one component on a line, with no force on it but the interactions given."""
    free = Interaction('free', roles={'target': 'p'}, laws=lambda s: [RateLaw(s.target.x.diff(s.t, 2), 0)])
    interactions = interactions or (free,)
    return System(elements=[BodyElement('p', x=[0.0])], interactions=list(interactions), space=ContinuousSpace(n_dim=1))


def test_equations_of_logistic_growth():
    """Test that logistic growth gives `dn/dt = r * n * (1 - n / K)`."""
    system = System(
        elements=[PopulationElement('p', x=10)],
        interactions=[LogisticGrowthInteraction('g', target='p')],
    )
    problem = system.prepare()._problem
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
    t = system.prepare()._problem.time
    assert bool(equation.lhs() == function('price_m')(t + 1))


def test_equations_of_group_with_pair_forces_are_of_order_two():
    """Test that a group with pair forces has one second order equation per component of its position."""
    system = LennardJonesFluidSystem(16).prepare().system_
    equations = symbolic(system).equations
    t = system.prepare()._problem.time
    assert list(equations) == ['fluid.x[i, 0]', 'fluid.x[i, 1]']
    assert bool(equations['fluid.x[i, 0]'].lhs() == diff(function('x_fluid_0')(t), t, 2))


def test_equations_of_jump_process_raises_error():
    """Test that jump processes have no symbolic equations."""
    system = System(
        elements=[AmountElement('I', x=1), AmountElement('R')],
        interactions=[RecoveryInteraction('r', infected='I', recovered='R', stochastic=True)],
    )
    with pytest.raises(AnalysisError, match='jump process'):
        symbolic(system)


def test_solve_symbolically_free_particle():
    """Test that a free particle moves linearly with two integration constants."""
    system = _free_particle()
    solution = symbolic(system, kind='solve').solutions['p.x[0]']
    t = system.prepare()._problem.time
    slope, start = solution.rhs().diff(t), solution.rhs().subs({t: 0})
    assert solution.rhs().diff(t, 2) == 0
    assert not slope.is_numeric()
    assert not start.is_numeric()
    assert (solution.rhs() - (slope * t + start)).simplify_full() == 0


def test_solve_symbolically_constant_acceleration():
    """Test that a constant force gives `F * t**2 / (2 * m)` from rest."""
    system = _free_particle(ConstantForceInteraction('push', target='p', F=3.0))
    problem = system.prepare()._problem
    F = problem.symbols['push.F']
    m = problem.symbols['p.m']
    solution = symbolic(system, kind='solve', use_initial_state=True).solutions['p.x[0]']
    assert (solution.rhs() - F * problem.time**2 / (2 * m)).simplify_full() == 0


def test_solve_symbolically_harmonic_oscillator():
    """Test that an oscillator from its initial state oscillates with frequency `sqrt(k / m)`."""
    system = System(
        elements=[BodyElement('p', x=[1.0], values={"x'": [0.0]})],
        interactions=[HarmonicPotentialInteraction('trap', target='p')],
        space=ContinuousSpace(n_dim=1),
    )
    problem = system.prepare()._problem
    k, m, t = problem.symbols['trap.k'], problem.symbols['p.m'], problem.time
    center = problem.symbols['trap.center']
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
    system = _free_particle(ConstantForceInteraction('push', target='p', F=float(force)))

    class SageForceInteraction(Interaction):
        """Force written as a Sage expression."""

        def __init__(self, label: str, *, target: Role) -> None:
            super().__init__(label)
            self.target = target

        def _prepare(self, context: SimpleNamespace) -> None:
            """Set an acceleration with a Sage value."""
            super()._prepare(context)
            self.laws_ = [RateLaw(context.target.x.diff(context.t, 2), [force])]

    sage_system = _free_particle(SageForceInteraction('push', target='p'))
    expected = system.prepare().simulate(until=1).get_values('p.x')
    result = sage_system.prepare().simulate(until=1).get_values('p.x')
    assert np.allclose(expected, result)


def _pendulum(rates, theta=0.1):
    """Return a pendulum `bob` of length `l` under the gravity `g`, whose angle follows rates of the namespace."""
    return _system(Element('bob', values={'theta': theta, "theta'": 0.0, 'g': 9.81, 'l': 1.0}), rates)


def _pendulum_rates(s):
    """Return the second derivative of the angle of a pendulum."""
    bob = s.target
    return [RateLaw(bob.theta.diff(s.t, 2), -(bob.g / bob.l) * sin(bob.theta))]


def _small_angle_rates(s):
    """Return the second derivative of the angle of a pendulum at small angles."""
    bob = s.target
    return [RateLaw(bob.theta.diff(s.t, 2), -(bob.g / bob.l) * bob.theta)]


def _first_order_pendulum():
    """Return a pendulum written as two first order equations of its angle and its angular velocity."""
    bob = Element('bob', values={'theta': 0.1, 'omega': 0.0, 'g': 9.81, 'l': 1.0})

    def rates(s):
        """Return the rates of the angle and of the angular velocity."""
        target = s.target
        return [RateLaw(target.theta, target.omega), RateLaw(target.omega, -(target.g / target.l) * sin(target.theta))]

    return _system(bob, rates)


def _third_order():
    """Return the third order equation of `x` whose third derivative is minus its first, from `sin(0)`."""
    element = Element('a', values={'x': 0.0, "x'": 1.0, "x''": 0.0})
    return _system(element, lambda s: [RateLaw(s.target.x.diff(s.t, 3), -s.target.x.diff(s.t))])


def _fibonacci():
    """Return the Fibonacci numbers, each the sum of the two before."""
    element = Element('f', values={'x': 1.0, 'x[-1]': 1.0})
    return _system(element, lambda s: [UpdateLaw(s.target.x, s.target.x + s.target.x(t=s.t - 1))])


def test_pendulum_equation_keeps_its_order():
    """Test that a pendulum of order 2 gives one second-order equation and no separate velocity equation."""
    equations = symbolic(_pendulum(_pendulum_rates)).equations
    assert list(equations) == ['bob.theta']
    theta = function('theta_bob')(SR.var('t'))
    g, l = SR.var('g_bob'), SR.var('l_bob')
    equation = equations['bob.theta']
    assert bool(equation.lhs() == diff(theta, SR.var('t'), 2))
    assert (equation.rhs() + g * sin(theta) / l).simplify_full() == 0


def test_lagged_equation_reads_past_values():
    """Test that a difference equation with a lag reads the value one step back."""
    equation = symbolic(_fibonacci()).equations['f.x']
    t = SR.var('t')
    x = function('x_f')
    assert bool(equation.lhs() == x(t + 1))
    assert (equation.rhs() - x(t) - x(t - 1)).simplify_full() == 0


def test_small_angle_pendulum_solved_with_plain_numbers():
    """Test that `θ'' = -(g / l) θ` with plain positive numbers solves to `0.1 cos(sqrt(g / l) t)`."""
    solution = symbolic(
        _pendulum(_small_angle_rates),
        kind='solve',
        use_initial_state=True,
    ).solutions['bob.theta']
    values = {SR.var('g_bob'): 9.81, SR.var('l_bob'): 2.0}
    for time in (0.3, 1.0, 2.5):
        expected = 0.1 * float(cos(sqrt(9.81 / 2.0) * time))
        assert float(solution.rhs().subs(values).subs({SR.var('t'): time})) == pytest.approx(expected)


def test_third_order_equation_solved():
    """Test that `x''' = -x'` from `x = 0`, `x' = 1`, `x'' = 0` solves to `sin(t)`."""
    solution = symbolic(
        _third_order(),
        kind='solve',
        use_initial_state=True,
    ).solutions['a.x']
    t = SR.var('t')
    for time in (0.3, 1.0, 2.5):
        assert float(solution.rhs().subs({t: time})) == pytest.approx(float(sin(time)))


@pytest.mark.parametrize(
    'system',
    [_pendulum(_pendulum_rates), _first_order_pendulum()],
    ids=['second-order', 'first-order'],
)
def test_nonlinear_pendulum_has_no_false_solution(system):
    """Test that the nonlinear pendulum raises the no-closed-form error instead of returning unevaluated transforms."""
    with pytest.raises(AnalysisError, match='No closed form was found'):
        symbolic(system, kind='solve', use_initial_state=True)


def test_remaining_sign_question_names_parameter():
    """Test that a sign Maxima still needs raises an error that names the parameter and the domain to declare."""

    spring = Element('a', values={'x': 0.1, "x'": 0.0, 'k': 0.0})
    system = _system(spring, lambda s: [RateLaw(s.target.x.diff(s.t, 2), -s.target.k * s.target.x)])
    with pytest.raises(AnalysisError, match=r'sign of k_a.*Solve the equations numerically'):
        symbolic(system, kind='solve', use_initial_state=True)


def test_symbolic_refuses_rewriting_systems_with_an_analysis_error():
    """Test that a system of words raises the analysis error of a form without equations, not a compiling error."""
    with pytest.raises(AnalysisError, match='rewriting system form, which has no equations'):
        symbolic(KochCurveSystem())


def test_hamiltonian_prints_operators_and_equations_refuse():
    """Test that an operator system prints its Hamiltonian and has no state equations."""
    system = QuantumHeisenbergSystem(n_sites=2, boundary='open').prepare().system_
    problem = QuantumCompiler().prepare(system).run().problem_
    hamiltonian = symbolic(problem, kind='hamiltonian').hamiltonian
    assert {'J_exchange', 'delta_exchange', 'h_field'} <= {str(variable) for variable in hamiltonian.variables()}
    assert 'local_operator(psi_chain_0, x)*local_operator(psi_chain_1, x)' in str(hamiltonian)
    with pytest.raises(AnalysisError, match='hamiltonian'):
        symbolic(problem)


def test_settings_the_kind_ignores_are_rejected():
    """Test that the initial state, which only solving reads, is rejected for the equations."""
    with pytest.raises(AnalysisError, match='does not read use_initial_state'):
        symbolic(LogisticGrowthSystem().prepare().system_, use_initial_state=True)


def test_mittag_leffler_exponential_case():
    """Test that E_(1, 1)(z) equals exp(z) within 1e-12."""
    z = np.linspace(-20.0, 5.0, 51)
    assert np.max(np.abs(_MittagLefflerFunction._get_mittag_leffler(1.0, 1.0, z) - np.exp(z))) < 1e-12


@pytest.mark.parametrize(
    ('alpha', 'beta', 'z', 'expected'),
    [
        (0.5, 1.0, -1.0, math.exp(1.0) * math.erfc(1.0)),
        (0.5, 1.0, -10.0, math.exp(100.0) * math.erfc(10.0)),
        (2.0, 1.0, -4.0, math.cos(2.0)),
        (2.0, 2.0, -4.0, math.sin(2.0) / 2.0),
        (1.0, 2.0, 1.5, math.expm1(1.5) / 1.5),
    ],
)
def test_mittag_leffler_closed_forms(alpha, beta, z, expected):
    """Test that the function matches its closed forms in error functions and trigonometric functions."""
    assert float(_MittagLefflerFunction._get_mittag_leffler(alpha, beta, z)) == pytest.approx(expected, rel=1e-12)


def test_mittag_leffler_asymptotic_matches_series():
    """Test that the asymptotic expansion for large arguments continues the series without a jump."""
    alpha = 0.6
    edge = -(50.0**alpha)
    below, above = _MittagLefflerFunction._get_mittag_leffler(alpha, 1.0, np.array([edge * 1.0001, edge * 0.9999]))
    assert below == pytest.approx(above, rel=1e-3)
    assert float(_MittagLefflerFunction._get_mittag_leffler(alpha, 1.0, -1e6)) == pytest.approx(
        1e-6 / math.gamma(1 - alpha),
        rel=1e-5,
    )


def test_mittag_leffler_complex_arguments():
    """Test that complex arguments give complex values, with E_(2, 1)(-z^2) = cos(z)."""
    value = _MittagLefflerFunction._get_mittag_leffler(2.0, 1.0, np.array([-((1.0 + 0.5j) ** 2)]))
    assert np.iscomplexobj(value)
    assert value[0] == pytest.approx(np.cos(1.0 + 0.5j), rel=1e-12)


def test_mittag_leffler_rejects_nonpositive_alpha():
    """Test that a nonpositive first parameter is rejected."""
    with pytest.raises(AnalysisError, match='alpha'):
        _MittagLefflerFunction._get_mittag_leffler(0.0, 1.0, 1.0)


def test_solve_symbolically_fractional_relaxation():
    """Test that linear fractional relaxation is solved by the Mittag-Leffler function of its order."""
    system = _relaxing()
    solution = symbolic(system, kind='solve', use_initial_state=True).solutions['r.x']
    assert str(solution) == 'x_r(t) == 2*mittag_leffler(3/5, 1, -lam_r*t^(3/5))'
    symbols = {str(item): item for item in solution.rhs().variables()}
    value = float(solution.rhs().subs({symbols['t']: 2.0, symbols['lam_r']: 0.5}))
    assert value == pytest.approx(
        2 * float(_MittagLefflerFunction._get_mittag_leffler(0.6, 1.0, -0.5 * 2.0**0.6)),
        rel=1e-12,
    )
    assert 'x_r(0)' in str(symbolic(system, kind='solve').solutions['r.x'])


def test_equations_of_fractional_state_write_caputo_derivative():
    """Test that the equation of a fractional state writes its Caputo derivative of the exact order."""
    assert str(symbolic(_relaxing()).equations['r.x']) == 'caputo(x_r(t), t, 3/5) == -lam_r*x_r(t)'


def test_solve_symbolically_rejects_nonlinear_fractional_law():
    """Test that a nonlinear fractional law has no exact solution."""
    with pytest.raises(AnalysisError, match='linear fractional relaxation'):
        symbolic(_saturating(), kind='solve')


def test_mittag_leffler_of_order_one_half_at_minus_one():
    """Test that `E_(1/2, 1)(-1)` is `exp(1) erfc(1)`, about 0.427584."""
    assert round(float(_MittagLefflerFunction._get_mittag_leffler(0.5, 1.0, -1.0)), 6) == 0.427584
