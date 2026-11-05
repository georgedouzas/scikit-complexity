"""Test the `_relaxation` module of `skcomplex.core.interactions`."""

import numpy as np
import pytest

from skcomplex.core.elements import GroupElement
from skcomplex.core.interactions import RelaxationInteraction
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import EulerMaruyamaSolver, ScipyODESolver
from skcomplex.errors import ValidationError

from ...conftest import HoldingElement, NodeElement

MEMBERS = 20_000


def test_run_reaches_the_ornstein_uhlenbeck_variance():
    """Test that goal 0, time `1 / theta`, and noise `sigma` give the variance `sigma**2 / (2 theta)` within 5%."""
    theta, sigma = 2.0, 0.6
    process = RelaxationInteraction('ou', target='walkers', state='n', goal=0.0, time=1 / theta, noise=sigma)
    system = System(
        elements=[GroupElement('walkers', element=HoldingElement, size=MEMBERS, values={'n': 0.0})],
        interactions=[process],
    )
    simulation = Simulation(system, solver=EulerMaruyamaSolver(dt=0.005), dt=0.5, random_state=0)
    values = simulation.prepare().run(until=6).trajectory_.values('walkers.n')[-4:]
    assert values.var(axis=1).mean() == pytest.approx(sigma**2 / (2 * theta), rel=0.05)
    assert values.mean() == pytest.approx(0.0, abs=0.01)


def test_run_relaxes_towards_a_goal_of_the_namespace():
    """Test that a relaxation towards the moving goal `t` follows `t - time (1 - exp(-t / time))`."""
    system = System(
        elements=[NodeElement('belief', x=0.0)],
        interactions=[RelaxationInteraction('adaptation', target='belief', goal=lambda s: s.t, time=2.0)],
    )
    solver = ScipyODESolver(rtol=1e-10, atol=1e-12)
    trajectory = Simulation(system, solver=solver, dt=0.5).prepare().run(until=3.0).trajectory_
    expected = trajectory.times - 2.0 * (1 - np.exp(-trajectory.times / 2.0))
    np.testing.assert_allclose(trajectory.values('belief.x'), expected, rtol=1e-7, atol=1e-9)


def test_check_rejects_a_goal_that_is_not_a_function():
    """Test that `check` rejects a goal that is neither callable nor a number."""
    with pytest.raises(ValidationError, match='goal'):
        RelaxationInteraction('ou', target='walkers', goal='zero').prepare()
