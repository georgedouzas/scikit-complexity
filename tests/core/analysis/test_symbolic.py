"""Test the symbolic analysis of systems."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR, cos, function, sqrt

from skcomplex.core.analysis import equations_of, solve_symbolically
from skcomplex.core.base import AnalysisError, role
from skcomplex.core.compile import compile_system
from skcomplex.core.model import ContinuousSpace, Interaction, Particle, Rate, System
from skcomplex.core.runtime import Simulation
from skcomplex.models.biology.ecology import LogisticGrowth, Population
from skcomplex.models.biology.epidemiology import Compartment, Recovery
from skcomplex.models.economics.finance import Asset, GeometricBrownianMotion
from skcomplex.models.economics.markets import Cobweb, Market
from skcomplex.models.physics.realizations import ConstantForce, HarmonicPotential
from tests.core.compile.test_compiler import CallableGrowth


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
