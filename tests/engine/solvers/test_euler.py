"""Test the explicit Euler integrator of ordinary differential equations."""

import numpy as np
import pytest

from skcomplex.core.elements import CompartmentElement
from skcomplex.core.interactions import RateInteraction, RelaxationInteraction
from skcomplex.core.laws.utils import delayed, step
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import EulerSolver
from skcomplex.errors import SolverError
from skcomplex.specs import StepSpec


def _relaxing_system():
    """Return a state that relaxes towards a goal."""
    relaxation = RelaxationInteraction('relaxation', target='tank', state='n', goal=10.0, time=2.0)
    return System(elements=[CompartmentElement('tank', n=0.0)], interactions=[relaxation])


def test_run_takes_explicit_steps():
    """Test that every step of a run is the state plus the step times the rate at the start of the step."""
    trajectory = Simulation(_relaxing_system(), solver=EulerSolver(dt=0.5), dt=0.5).check().prepare().run(3).trajectory_
    expected = [0.0]
    for _ in range(6):
        expected.append(expected[-1] + 0.5 * (10.0 - expected[-1]) / 2.0)
    assert np.allclose(trajectory.values('tank.n'), expected)


def test_run_splits_a_frame_into_steps():
    """Test that a frame longer than the step takes several steps."""
    system = _relaxing_system()
    frames = Simulation(system, solver=EulerSolver(dt=0.5), dt=1.0).check().prepare().run(3).trajectory_
    steps = Simulation(system, solver=EulerSolver(dt=0.5), dt=0.5).check().prepare().run(3).trajectory_
    assert np.allclose(frames.values('tank.n'), steps.values('tank.n')[::2])


def test_run_rejects_a_step_that_is_not_positive():
    """Test that a step that is not positive raises an error."""
    problem = compile_system(_relaxing_system())
    stepper = EulerSolver(dt=0.0, random_state=0).check().prepare(problem)
    with pytest.raises(SolverError, match='must be positive'):
        stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0))


def _delayed_rate(s):
    """Return the value of the state one time unit ago."""
    return delayed(s.target.n, 1.0)


def _stepped_rate(s):
    """Return a rate that rises at a time."""
    return step(s.t, 1.0, 2.0)


def test_run_reads_delays_from_the_steps_of_the_run():
    """Test that a delayed value reads the state of the step that the delay names, the first state before the start."""
    system = System(
        elements=[CompartmentElement('tank', n=1.0)],
        interactions=[RateInteraction('rate', target='tank', state='n', rate=_delayed_rate)],
    )
    trajectory = Simulation(system, solver=EulerSolver(dt=0.25), dt=0.25).check().prepare().run(2).trajectory_
    expected = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5625, 2.9375, 3.375]
    assert np.allclose(trajectory.values('tank.n'), expected)


def test_run_steps_over_a_signal():
    """Test that a signal of time takes its new value at the step that starts at its time."""
    system = System(
        elements=[CompartmentElement('tank', n=0.0)],
        interactions=[RateInteraction('rate', target='tank', state='n', rate=_stepped_rate)],
    )
    trajectory = Simulation(system, solver=EulerSolver(dt=0.25), dt=0.25).check().prepare().run(2).trajectory_
    assert np.allclose(trajectory.values('tank.n'), [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0])
