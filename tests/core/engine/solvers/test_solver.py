"""Test the `Solver` class."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import NoiseLaw, RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile import compile_system
from skcomplex.core.engine.problems._steps import step_count
from skcomplex.core.engine.solvers import (
    EulerMaruyamaSolver,
    GillespieSolver,
    HeunSolver,
    IterationSolver,
    MilsteinSolver,
    RK4Solver,
    VelocityVerletSolver,
)
from skcomplex.core.errors import SolverError
from skcomplex.core.specs import StateSpec, StepSpec, VisualSpec
from skcomplex.systems.biology.common.populations import ExponentialGrowthInteraction, PopulationElement
from skcomplex.systems.economics.markets import CobwebInteraction, MarketElement


class BeadElement(Element):
    """Bead element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 1.0,
        v: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v

    def _get_laws(self, s):
        """Return the laws of the bead element."""
        return [RateLaw(s.x, s.v), RateLaw(s.v, -s.x**3)]


class WalkerElement(Element):
    """Walker element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x

    def _get_laws(self, s):
        """Return the laws of the walker element."""
        return [RateLaw(s.x, -s.x**3), NoiseLaw(s.x, 0.5 * s.x)]


def _growth_problem():
    """Return the compiled problem of exponential growth."""
    system = System(
        elements=[PopulationElement('a', n=1)],
        interactions=[ExponentialGrowthInteraction('g', target='a', r=1.0)],
    )
    return compile_system(system)


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


def _system(kind):
    """Return a nonlinear system of the oscillator, noise, or map kind."""
    if kind == 'ode':
        return System(elements=[GroupElement('beads', element=BeadElement, size=3, values={'x': [1.0, 0.5, -0.2]})])
    if kind == 'sde':
        return System(
            elements=[GroupElement('walkers', element=WalkerElement, size=3, values={'x': [1.0, 0.5, -0.2]})],
        )
    return System(
        elements=[MarketElement('m', price=1.0)],
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
    for backend in (NumpyBackend(), NumbaBackend()):
        problem = compile_system(_system(kind), backend=backend)
        rng = np.random.default_rng(0)
        stepper.set_params(random_state=rng).prepare(problem)
        states.append(stepper.run(StepSpec(0.1, problem.initial_state, problem.initial_parameters, 3.4)).state_)
        generators.append(rng.bit_generator.state)
    np.testing.assert_allclose(states[0], states[1], rtol=1e-12)
    assert generators[0] == generators[1]
