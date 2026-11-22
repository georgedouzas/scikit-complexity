"""Test the `Solver` class."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import NoiseLaw, RateLaw
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.problems._steps import step_count
from skcomplex.core.engine.solvers import (
    EulerMaruyamaSolver,
    GillespieSolver,
    HeunSolver,
    IterationSolver,
    MilsteinSolver,
    RK4Solver,
    StepSpec,
    VelocityVerletSolver,
)
from skcomplex.core.errors import SolverError
from skcomplex.core.systems import System
from skcomplex.systems.biology.common.populations import ExponentialGrowthInteraction, PopulationElement
from skcomplex.systems.economics.markets import CobwebInteraction, MarketElement


def _growth_problem():
    """Return the compiled problem of exponential growth."""
    system = System(
        elements=[PopulationElement('a', x=1)],
        interactions=[ExponentialGrowthInteraction('g', target='a', r=1.0)],
    )
    return system.prepare()._problem


def test_stepper_wrong_form_raises_error():
    """Test that a stepper refuses a problem of another form."""
    with pytest.raises(SolverError, match='jump process form'):
        GillespieSolver(random_state=0).prepare(_growth_problem())


@pytest.mark.parametrize(
    ('t', 'until', 'dt', 'count'),
    [(0.1, 0.4, 0.1, 3), (0.0, 1.0, 0.3, 4), (0.0, 1e-20, 0.1, 1), (0.0, 1.0, 1.0, 1), (0.0, 1.0 + 1e-12, 0.1, 10)],
)
def test_step_count_rounds_up_beyond_rounding(t, until, dt, count):
    """Test that the step count rounds up, takes at least one step, and ignores a span just above whole steps."""
    assert step_count(t, until, dt) == count


def _bead_laws(s):
    """Return the laws of the beads, `x' = v` and `v' = -x^3`."""
    x, v = s.target.x, s.target.v
    return [RateLaw(x.diff(s.t), v), RateLaw(v.diff(s.t), -(x**3))]


def _walker_laws(s):
    """Return the laws of the walkers, a cubic drift and a multiplicative noise."""
    x = s.target.x
    return [RateLaw(x.diff(s.t), -(x**3)), NoiseLaw(x, 0.5 * x)]


def _system(kind, backend):
    """Return a nonlinear system of the oscillator, noise, or map kind on a backend."""
    start = [1.0, 0.5, -0.2]
    if kind == 'ode':
        beads = Element('beads', values={'x': start, 'v': [0.0, 0.0, 0.0]}, size=3)
        motion = Interaction('motion', roles={'target': 'beads'}, laws=_bead_laws)
        return System(elements=[beads], interactions=[motion], backend=backend)
    if kind == 'sde':
        walkers = Element('walkers', values={'x': start}, size=3)
        motion = Interaction('motion', roles={'target': 'walkers'}, laws=_walker_laws)
        return System(elements=[walkers], interactions=[motion], backend=backend)
    return System(
        elements=[MarketElement('m', price=1.0)],
        interactions=[CobwebInteraction('c', target='m', a=10, b=2, c=1, d=1)],
        backend=backend,
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
    for backend in (NumpyBackend(), NumbaBackend()):
        problem = _system(kind, backend).prepare()._problem
        rng = np.random.default_rng(0)
        stepper.set_params(random_state=rng).prepare(problem)
        states.append(stepper.run(StepSpec(0.1, problem.initial_state, problem.initial_parameters, 3.4)).state_)
        generators.append(rng.bit_generator.state)
    np.testing.assert_allclose(states[0], states[1], rtol=1e-12)
    assert generators[0] == generators[1]
