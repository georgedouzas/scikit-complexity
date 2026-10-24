"""Test the symbolic analysis of systems."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR, cos, diff, function, sin, sqrt

from skcomplex.core.base import AnalysisError, role, state
from skcomplex.core.components import Element
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import Rate, Update, derivative, lag
from skcomplex.core.space import ContinuousSpace
from skcomplex.core.system import System
from skcomplex.engine.analysis import equations_of, solve_symbolically
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.formalisms.particles import Particle
from skcomplex.models.biology.ecology import LogisticGrowth, Population
from skcomplex.models.biology.epidemiology import Compartment, Recovery
from skcomplex.models.economics.finance import Asset, GeometricBrownianMotion
from skcomplex.models.economics.markets import Cobweb, Market
from skcomplex.models.physics.realizations.particles import ConstantForce, HarmonicPotential

from ...conftest import CallableGrowth


def test_equations_of_logistic_growth():
    """Test that logistic growth gives `dn/dt = r * n * (1 - n / K)`."""
    system = System(components=[Population('p', n=10)], interactions=[LogisticGrowth('g', target='p')])
    problem = compile_system(system)
    n = function('n_p')(problem.time)
    r, K = problem.symbols['g.r'], problem.symbols['g.K']
    equation = equations_of(system)['p.n']
    assert equation.is_relational()
    assert bool(equation.lhs() == n.diff(problem.time))
    assert (equation.rhs() - r * n * (1 - n / K)).simplify_full() == 0


def test_equations_of_discrete_map_gives_next_value():
    """Test that a discrete map gives the value at `t + 1`."""
    system = System(components=[Market('m')], interactions=[Cobweb('c', target='m')])
    equation = equations_of(system)['m.price']
    t = compile_system(system).time
    assert bool(equation.lhs() == function('price_m')(t + 1))


def test_equations_of_jump_process_raises_error():
    """Test that jump processes have no symbolic equations."""
    system = System(
        components=[Compartment('I', n=1), Compartment('R')],
        interactions=[Recovery('r', infected='I', recovered='R', stochastic=True)],
    )
    with pytest.raises(AnalysisError, match='jump process'):
        equations_of(system)


def test_solve_symbolically_free_particle():
    """Test that a free particle moves linearly with two integration constants."""
    system = System(components=[Particle('p')], space=ContinuousSpace(n_dim=1))
    solution = solve_symbolically(system)['p.x[0]']
    t = compile_system(system).time
    slope, start = solution.rhs().diff(t), solution.rhs().subs({t: 0})
    assert solution.rhs().diff(t, 2) == 0
    assert not slope.is_numeric()
    assert not start.is_numeric()
    assert (solution.rhs() - (slope * t + start)).simplify_full() == 0


def test_solve_symbolically_constant_acceleration():
    """Test that a constant force gives `F * t**2 / (2 * m)` from rest."""
    system = System(
        components=[Particle('p')],
        interactions=[ConstantForce('push', target='p', F=3.0)],
        space=ContinuousSpace(n_dim=1),
    )
    problem = compile_system(system)
    (F,) = problem.symbols['push.F']
    m = problem.symbols['p.m']
    solution = solve_symbolically(system, use_initial_state=True)['p.x[0]']
    assert (solution.rhs() - F * problem.time**2 / (2 * m)).simplify_full() == 0


def test_solve_symbolically_harmonic_oscillator():
    """Test that an oscillator from its initial state oscillates with frequency `sqrt(k / m)`."""
    system = System(
        components=[Particle('p', x=1.0, v=0.0)],
        interactions=[HarmonicPotential('trap', target='p')],
        space=ContinuousSpace(n_dim=1),
    )
    problem = compile_system(system)
    k, m, t = problem.symbols['trap.k'], problem.symbols['p.m'], problem.time
    (center,) = problem.symbols['trap.center']
    solution = solve_symbolically(system, use_initial_state=True)['p.x[0]'].rhs().subs({center: 0})
    expected = cos(sqrt(k / m) * t)
    values = {k: 2.0, m: 0.5}
    for time in np.linspace(0, 5, 6):
        assert float(solution.subs(values).subs({t: time})) == pytest.approx(
            float(expected.subs(values).subs({t: time})),
        )


def test_solve_symbolically_not_ode_raises_error():
    """Test that only ODE systems are solved."""
    system = System(components=[Asset('s')], interactions=[GeometricBrownianMotion('g', target='s')])
    with pytest.raises(AnalysisError, match='stochastic differential equations'):
        solve_symbolically(system)


def test_solve_symbolically_callable_raises_error():
    """Test that callable laws cannot be solved."""
    system = System(components=[Population('rabbits')], interactions=[CallableGrowth('g', target='rabbits')])
    with pytest.raises(AnalysisError, match='callable'):
        solve_symbolically(system)


def test_sage_expression_compiles_like_number():
    """Test that a constant force given as a Sage expression simulates like the same force given as a number."""
    force = SR(2) * SR(1.5)
    system = System(
        components=[Particle('p')],
        interactions=[ConstantForce('push', target='p', F=float(force))],
        space=ContinuousSpace(n_dim=1),
    )

    class SageForce(Interaction):
        """Force written as a Sage expression."""

        target: str = role()

        def laws(self, s):
            """Return a rate with a Sage value."""
            return [Rate(s.target.v, [force])]

    sage_system = System(
        components=[Particle('p')],
        interactions=[SageForce('push', target='p')],
        space=ContinuousSpace(n_dim=1),
    )
    expected = Simulation(system).run(until=1).trajectory_['p.x']
    result = Simulation(sage_system).run(until=1).trajectory_['p.x']
    assert np.allclose(expected, result)


Pendulum = Element.define(
    'Pendulum',
    states={'theta': state(default=0.1, order=2)},
    parameters={'g': 9.81, 'l': 1.0},
    laws=lambda s: [Rate(s.theta, -(s.g / s.l) * sin(s.theta))],
)


SmallAngle = Element.define(
    'SmallAngle',
    states={'theta': state(default=0.1, order=2)},
    parameters={'g': 9.81, 'l': 1.0},
    laws=lambda s: [Rate(s.theta, -(s.g / s.l) * s.theta)],
)


ThirdOrder = Element.define(
    'ThirdOrder',
    states={'x': state(default=0.0, order=3, initial=(1.0, 0.0))},
    laws=lambda s: [Rate(s.x, -derivative(s.x, 1))],
)


FirstOrderPendulum = Element.define(
    'FirstOrderPendulum',
    states={'theta': 0.1, 'omega': 0.0},
    parameters={'g': 9.81, 'l': 1.0},
    laws=lambda s: [Rate(s.theta, s.omega), Rate(s.omega, -(s.g / s.l) * sin(s.theta))],
)


Fibonacci = Element.define(
    'Fibonacci',
    states={'x': state(default=1.0, lags=1, initial=(1.0,))},
    laws=lambda s: [Update(s.x, s.x + lag(s.x, 1))],
)


def test_pendulum_equation_keeps_its_order():
    """Test that a pendulum of order 2 gives one second-order equation and no separate velocity equation."""
    equations = equations_of(System(components=[Pendulum('bob')]))
    assert list(equations) == ['bob.theta']
    theta = function('theta_bob')(SR.var('t'))
    g, l = SR.var('g_bob'), SR.var('l_bob')
    equation = equations['bob.theta']
    assert bool(equation.lhs() == diff(theta, SR.var('t'), 2))
    assert (equation.rhs() + g * sin(theta) / l).simplify_full() == 0


def test_lagged_equation_reads_past_values():
    """Test that a difference equation with a lag reads the value one step back."""
    equation = equations_of(System(components=[Fibonacci('f')]))['f.x']
    t = SR.var('t')
    x = function('x_f')
    assert bool(equation.lhs() == x(t + 1))
    assert (equation.rhs() - x(t) - x(t - 1)).simplify_full() == 0


def test_small_angle_pendulum_solved_with_plain_numbers():
    """Test that `θ'' = -(g / l) θ` with plain positive numbers solves to `0.1 cos(sqrt(g / l) t)`."""
    solution = solve_symbolically(System(components=[SmallAngle('bob')]), use_initial_state=True)['bob.theta']
    values = {SR.var('g_bob'): 9.81, SR.var('l_bob'): 2.0}
    for time in (0.3, 1.0, 2.5):
        expected = 0.1 * float(cos(sqrt(9.81 / 2.0) * time))
        assert float(solution.rhs().subs(values).subs({SR.var('t'): time})) == pytest.approx(expected)


def test_third_order_equation_solved():
    """Test that `x''' = -x'` from `x = 0`, `x' = 1`, `x'' = 0` solves to `sin(t)`."""
    solution = solve_symbolically(System(components=[ThirdOrder('a')]), use_initial_state=True)['a.x']
    t = SR.var('t')
    for time in (0.3, 1.0, 2.5):
        assert float(solution.rhs().subs({t: time})) == pytest.approx(float(sin(time)))


@pytest.mark.parametrize('element', [Pendulum('bob'), FirstOrderPendulum('bob')], ids=['second-order', 'first-order'])
def test_nonlinear_pendulum_has_no_false_solution(element):
    """Test that the nonlinear pendulum raises the no-closed-form error instead of returning unevaluated transforms."""
    with pytest.raises(AnalysisError, match='No closed form was found'):
        solve_symbolically(System(components=[element]), use_initial_state=True)


def test_remaining_sign_question_names_parameter():
    """Test that a sign Maxima still needs raises an error that names the parameter and the domain to declare."""
    Spring = Element.define(
        'Spring',
        states={'x': state(default=0.1, order=2)},
        parameters={'k': 0.0},
        laws=lambda s: [Rate(s.x, -s.k * s.x)],
    )
    with pytest.raises(AnalysisError, match=r'sign of k_a.*domain=Domain\.POSITIVE'):
        solve_symbolically(System(components=[Spring('a')]), use_initial_state=True)
