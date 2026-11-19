"""Test the explicit Euler integrator of ordinary differential equations."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import step
from skcomplex.core.engine.solvers import EulerSolver, StepSpec
from skcomplex.core.errors import SolverError
from skcomplex.core.systems import System


class AmountElement(Element):
    """Element that holds one nonnegative scalar `x`, an amount."""

    def __init__(self, label: str, *, x: Annotated[float, RealSet.closed_open(0, oo)] = 0.0) -> None:
        super().__init__(label)
        self.x = x


def _rate_system(rate, x: float = 0.0, dt: float = 0.5) -> System:
    """Return a tank whose amount changes at a rate, a function of the namespace, run by Euler steps of `dt`."""
    interaction = Interaction('rate', roles={'target': 'tank'}, laws=lambda s: [RateLaw(s.target.x.diff(s.t), rate(s))])
    return System(elements=[AmountElement('tank', x=x)], interactions=[interaction], solver=EulerSolver(dt=dt))


def _relaxing_system() -> System:
    """Return a state that relaxes towards a goal."""
    return _rate_system(lambda s: (10.0 - s.target.x) / 2.0)


def test_run_takes_explicit_steps():
    """Test that every step of a run is the state plus the step times the rate at the start of the step."""
    trajectory = _relaxing_system().prepare().simulate(until=3, dt=0.5)
    expected = [0.0]
    for _ in range(6):
        expected.append(expected[-1] + 0.5 * (10.0 - expected[-1]) / 2.0)
    assert np.allclose(trajectory.get_values('tank.x'), expected)


def test_run_splits_a_frame_into_steps():
    """Test that a frame longer than the step takes several steps."""
    system = _relaxing_system().prepare()
    frames = system.simulate(until=3, dt=1.0)
    steps = system.simulate(until=3, dt=0.5)
    assert np.allclose(frames.get_values('tank.x'), steps.get_values('tank.x')[::2])


def test_run_rejects_a_step_that_is_not_positive():
    """Test that a step that is not positive raises an error."""
    problem = _relaxing_system().prepare()._problem
    stepper = EulerSolver(dt=0.0, random_state=0).prepare(problem)
    with pytest.raises(SolverError, match='must be positive'):
        stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0))


def _delayed_rate(s):
    """Return the value of the state one time unit ago."""
    return s.target.x(t=s.t - 1.0)


def _stepped_rate(s):
    """Return a rate that rises at a time."""
    return step(s.t, 1.0, 2.0)


def test_run_reads_delays_from_the_steps_of_the_run():
    """Test that a delayed value reads the state of the step that the delay names, the first state before the start."""
    trajectory = _rate_system(_delayed_rate, x=1.0, dt=0.25).prepare().simulate(until=2, dt=0.25)
    expected = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5625, 2.9375, 3.375]
    assert np.allclose(trajectory.get_values('tank.x'), expected)


def test_run_steps_over_a_signal():
    """Test that a signal of time takes its new value at the step that starts at its time."""
    trajectory = _rate_system(_stepped_rate, dt=0.25).prepare().simulate(until=2, dt=0.25)
    assert np.allclose(trajectory.get_values('tank.x'), [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0])
