"""Test the stepper contract, the count of fixed steps, the solver registry, and the parity of backends."""

import numpy as np
import pytest

from skcomplex.core.base import SolverError
from skcomplex.core.components import Element
from skcomplex.core.laws import Noise, Rate
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.solvers import (
    RK4,
    EulerMaruyama,
    Gillespie,
    Heun,
    Iteration,
    Milstein,
    ScipyODE,
    VelocityVerlet,
    resolve_solver,
    step_count,
)
from skcomplex.models.biology.ecology import ExponentialGrowth, Population
from skcomplex.models.economics.markets import Cobweb, Market

Bead = Element.define('Bead', states={'x': 1.0, 'v': 0.0}, laws=lambda s: [Rate(s.x, s.v), Rate(s.v, -s.x**3)])
Walker = Element.define('Walker', states={'x': 1.0}, laws=lambda s: [Rate(s.x, -s.x**3), Noise(s.x, 0.5 * s.x)])


def _growth_problem():
    """Return the compiled problem of exponential growth."""
    system = System(components=[Population('a', n=1)], interactions=[ExponentialGrowth('g', target='a', r=1.0)])
    return compile_system(system)


def test_stepper_wrong_form_raises_error():
    """Test that a stepper refuses a problem of another form."""
    with pytest.raises(SolverError, match='jump process form'):
        Gillespie().initialize(_growth_problem(), np.random.default_rng(0))


def test_resolve_solver_by_name_and_instance():
    """Test that solvers resolve from names, the default, and instances."""
    problem = _growth_problem()
    assert isinstance(resolve_solver(problem, 'auto'), ScipyODE)
    assert isinstance(resolve_solver(problem, 'scipy'), ScipyODE)
    stepper = ScipyODE()
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
        return System(components=[Bead.group('beads', 3, x=[1.0, 0.5, -0.2])])
    if kind == 'sde':
        return System(components=[Walker.group('walkers', 3, x=[1.0, 0.5, -0.2])])
    return System(components=[Market('m', price=1.0)], interactions=[Cobweb('c', target='m', a=10, b=2, c=1, d=1)])


@pytest.mark.parametrize(
    ('stepper', 'kind'),
    [
        (RK4(dt=0.1), 'ode'),
        (VelocityVerlet(dt=0.1), 'ode'),
        (EulerMaruyama(dt=0.1), 'sde'),
        (Milstein(dt=0.1), 'sde'),
        (Heun(dt=0.1), 'sde'),
        (Iteration(), 'map'),
    ],
)
def test_fixed_step_steppers_agree_on_both_backends(stepper, kind):
    """Test that every fixed-step stepper takes the same steps and draws on the NumPy and Numba backends."""
    states, generators = [], []
    for backend in ('numpy', 'numba'):
        problem = compile_system(_system(kind), backend=backend)
        rng = np.random.default_rng(0)
        stepper.initialize(problem, rng)
        states.append(stepper.advance(0.1, problem.initial_state, problem.initial_parameters, 3.4))
        generators.append(rng.bit_generator.state)
    np.testing.assert_allclose(states[0], states[1], rtol=1e-12)
    assert generators[0] == generators[1]
