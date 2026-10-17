"""Test the symbolic analysis of systems."""

import numpy as np
import pytest
import sympy

from skcomplex.analysis import equations_of, solve_symbolically
from skcomplex.compile import compile_system
from skcomplex.core import AnalysisError, role
from skcomplex.model import EuclideanSpace, Interaction, Rate, System
from skcomplex.models.ecology import LogisticGrowth, Population
from skcomplex.models.economics import Cobweb, Market
from skcomplex.models.epidemiology import Compartment, Recovery
from skcomplex.models.finance import Asset, GeometricBrownianMotion
from skcomplex.models.physics import ConstantForce, HarmonicPotential, Particle
from skcomplex.runtime import Simulation
from tests.compile.test_compiler import CallableGrowth


def test_equations_of_logistic_growth():
    """Test that logistic growth gives `dn/dt = r * n * (1 - n / K)`."""
    system = System(elements=[Population('p', n=10)], interactions=[LogisticGrowth('g', target='p')])
    problem = compile_system(system)
    n = sympy.Function('n_p')(problem.time)
    r, K = problem.symbols['g.r'], problem.symbols['g.K']
    equation = equations_of(system)['p.n']
    assert equation.lhs == sympy.Derivative(n, problem.time)
    assert sympy.simplify(equation.rhs - r * n * (1 - n / K)) == 0


def test_equations_of_discrete_map_gives_next_value():
    """Test that a discrete map gives the value at `t + 1`."""
    system = System(elements=[Market('m')], interactions=[Cobweb('c', target='m')])
    equation = equations_of(system)['m.price']
    t = compile_system(system).time
    assert equation.lhs == sympy.Function('price_m')(t + 1)


def test_equations_of_jump_process_raises_error():
    """Test that jump processes have no symbolic equations."""
    system = System(
        elements=[Compartment('I', n=1), Compartment('R')],
        interactions=[Recovery('r', infected='I', recovered='R', stochastic=True)],
    )
    with pytest.raises(AnalysisError, match='jump process'):
        equations_of(system)


def test_solve_symbolically_free_particle():
    """Test that a free particle moves linearly with two integration constants."""
    system = System(elements=[Particle('p')], space=EuclideanSpace(1))
    solution = solve_symbolically(system)['p.x[0]']
    t = compile_system(system).time
    constants = solution.rhs.free_symbols - {t}
    assert len(constants) == 2
    assert sympy.diff(solution.rhs, t, 2) == 0


def test_solve_symbolically_constant_acceleration():
    """Test that a constant force gives `F * t**2 / (2 * m)` from rest."""
    system = System(
        elements=[Particle('p')],
        interactions=[ConstantForce('push', target='p', F=3.0)],
        space=EuclideanSpace(1),
    )
    problem = compile_system(system)
    (F,) = problem.symbols['push.F']
    m = problem.symbols['p.m']
    solution = solve_symbolically(system, use_initial_state=True)['p.x[0]']
    assert sympy.simplify(solution.rhs - F * problem.time**2 / (2 * m)) == 0


def test_solve_symbolically_harmonic_oscillator():
    """Test that an oscillator from its initial state oscillates with frequency `sqrt(k / m)`."""
    system = System(
        elements=[Particle('p', x=1.0, v=0.0)],
        interactions=[HarmonicPotential('trap', target='p')],
        space=EuclideanSpace(1),
    )
    problem = compile_system(system)
    k, m, t = problem.symbols['trap.k'], problem.symbols['p.m'], problem.time
    (center,) = problem.symbols['trap.center']
    solution = solve_symbolically(system, use_initial_state=True)['p.x[0]'].rhs.subs(center, 0)
    expected = sympy.cos(sympy.sqrt(k / m) * t)
    values = {k: 2.0, m: 0.5}
    for time in np.linspace(0, 5, 6):
        assert float(solution.subs(values).subs(t, time)) == pytest.approx(float(expected.subs(values).subs(t, time)))


def test_solve_symbolically_not_ode_raises_error():
    """Test that only ODE systems are solved."""
    system = System(elements=[Asset('s')], interactions=[GeometricBrownianMotion('g', target='s')])
    with pytest.raises(AnalysisError, match='stochastic differential equations'):
        solve_symbolically(system)


def test_solve_symbolically_callable_raises_error():
    """Test that callable laws cannot be solved."""
    system = System(elements=[Population('rabbits')], interactions=[CallableGrowth('g', target='rabbits')])
    with pytest.raises(AnalysisError, match='callable'):
        solve_symbolically(system)


def test_sage_expression_compiles_like_sympy():
    """Test that a Sage force simulates like the equivalent SymPy force."""
    sage = pytest.importorskip('sage.all__sagemath_symbolics')
    force = sage.SR(2) * sage.SR(1.5)
    system = System(
        elements=[Particle('p')],
        interactions=[ConstantForce('push', target='p', F=float(force))],
        space=EuclideanSpace(1),
    )

    class SageForce(Interaction):
        """Force written as a Sage expression."""

        target: str = role()

        def laws(self, s):
            """Return a rate with a Sage value."""
            return [Rate(s.target.v, [force])]

    sage_system = System(
        elements=[Particle('p')],
        interactions=[SageForce('push', target='p')],
        space=EuclideanSpace(1),
    )
    expected = Simulation(system).run(until=1).trajectory_['p.x']
    result = Simulation(sage_system).run(until=1).trajectory_['p.x']
    assert np.allclose(expected, result)
