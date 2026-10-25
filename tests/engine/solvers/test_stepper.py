"""Test the stepper contract, the count of fixed steps, the solver registry, and the parity of backends."""

import numpy as np
import pytest

from skcomplex.core.base import SolverError
from skcomplex.core.components import ElementComponent
from skcomplex.core.laws import NoiseLaw, RateLaw
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.solvers import (
    EulerMaruyamaSolver,
    GillespieSolver,
    HeunSolver,
    IterationSolver,
    MilsteinSolver,
    RK4Solver,
    ScipyODESolver,
    VelocityVerletSolver,
    resolve_solver,
    step_count,
)
from skcomplex.models.biology.ecology import ExponentialGrowthInteraction, PopulationComponent
from skcomplex.models.economics.markets import CobwebInteraction, MarketComponent

BeadComponent = ElementComponent.define(
    'BeadComponent',
    states={'x': 1.0, 'v': 0.0},
    laws=lambda s: [RateLaw(s.x, s.v), RateLaw(s.v, -s.x**3)],
)
WalkerComponent = ElementComponent.define(
    'WalkerComponent',
    states={'x': 1.0},
    laws=lambda s: [RateLaw(s.x, -s.x**3), NoiseLaw(s.x, 0.5 * s.x)],
)


def _growth_problem():
    """Return the compiled problem of exponential growth."""
    system = System(
        components=[PopulationComponent('a', n=1)],
        interactions=[ExponentialGrowthInteraction('g', target='a', r=1.0)],
    )
    return compile_system(system)


def test_stepper_wrong_form_raises_error():
    """Test that a stepper refuses a problem of another form."""
    with pytest.raises(SolverError, match='jump process form'):
        GillespieSolver().fit(_growth_problem(), np.random.default_rng(0))


def test_resolve_solver_by_name_and_instance():
    """Test that solvers resolve from names, the default, and instances."""
    problem = _growth_problem()
    assert isinstance(resolve_solver(problem, 'auto'), ScipyODESolver)
    assert isinstance(resolve_solver(problem, 'scipy'), ScipyODESolver)
    stepper = ScipyODESolver()
    assert resolve_solver(problem, stepper) is stepper
    with pytest.raises(SolverError, match='Available names are'):
        resolve_solver(problem, 'gillespie')


@pytest.mark.parametrize(
    ('t', 'until', 'dt', 'count'),
    [(0.1, 0.4, 0.1, 3), (0.0, 1.0, 0.3, 4), (0.0, 1e-20, 0.1, 1), (0.0, 1.0, 1.0, 1), (0.0, 1.0 + 1e-12, 0.1, 10)],
)
def test_step_count_rounds_up_beyond_rounding(t, until, dt, count):
    """Test that the step count rounds up, takes at least one step, and ignores a span just above whole steps."""
    assert step_count(t, until, dt) == count


def _system(kind):
    """Return a nonlinear system of the oscillator, noise, or map kind."""
    if kind == 'ode':
        return System(components=[BeadComponent.group('beads', 3, x=[1.0, 0.5, -0.2])])
    if kind == 'sde':
        return System(components=[WalkerComponent.group('walkers', 3, x=[1.0, 0.5, -0.2])])
    return System(
        components=[MarketComponent('m', price=1.0)],
        interactions=[CobwebInteraction('c', target='m', a=10, b=2, c=1, d=1)],
    )


@pytest.mark.parametrize(
    ('stepper', 'kind'),
    [
        (RK4Solver(dt=0.1), 'ode'),
        (VelocityVerletSolver(dt=0.1), 'ode'),
        (EulerMaruyamaSolver(dt=0.1), 'sde'),
        (MilsteinSolver(dt=0.1), 'sde'),
        (HeunSolver(dt=0.1), 'sde'),
        (IterationSolver(), 'map'),
    ],
)
def test_fixed_step_steppers_agree_on_both_backends(stepper, kind):
    """Test that every fixed-step stepper takes the same steps and draws on the NumPy and Numba backends."""
    states, generators = [], []
    for backend in ('numpy', 'numba'):
        problem = compile_system(_system(kind), backend=backend)
        rng = np.random.default_rng(0)
        stepper.fit(problem, rng)
        states.append(stepper.run(0.1, problem.initial_state, problem.initial_parameters, 3.4))
        generators.append(rng.bit_generator.state)
    np.testing.assert_allclose(states[0], states[1], rtol=1e-12)
    assert generators[0] == generators[1]
