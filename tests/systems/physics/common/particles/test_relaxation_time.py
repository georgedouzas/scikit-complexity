"""Test the `_relaxation_time` module of `skcomplex.systems.physics.common.particles`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.engine.solvers import EulerMaruyamaSolver, ScipyODESolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.particles import RelaxationTimeInteraction

MEMBERS = 20_000


def test_run_reaches_the_ornstein_uhlenbeck_variance():
    """Test that goal 0, time `1 / theta`, and noise `sigma` give the variance `sigma**2 / (2 theta)` within 5%."""
    theta, sigma = 2.0, 0.6
    process = RelaxationTimeInteraction('ou', target='walkers', state='n', goal=0.0, time=1 / theta, noise=sigma)
    system = System(
        elements=[Element('walkers', values={'n': np.zeros(MEMBERS)}, size=MEMBERS)],
        interactions=[process],
        solver=EulerMaruyamaSolver(dt=0.005),
    )
    values = system.prepare().simulate(until=6, dt=0.5, random_state=0).get_values('walkers.n')[-4:]
    assert values.var(axis=1).mean() == pytest.approx(sigma**2 / (2 * theta), rel=0.05)
    assert values.mean() == pytest.approx(0.0, abs=0.01)


def test_run_relaxes_towards_a_goal_of_the_namespace():
    """Test that a relaxation towards the moving goal `t` follows `t - time (1 - exp(-t / time))`."""
    system = System(
        elements=[Element('belief', values={'x': 0.0})],
        interactions=[RelaxationTimeInteraction('adaptation', target='belief', goal=lambda s: s.t, time=2.0)],
        solver=ScipyODESolver(rtol=1e-10, atol=1e-12),
    )
    trajectory = system.prepare().simulate(until=3.0, dt=0.5)
    expected = trajectory.times - 2.0 * (1 - np.exp(-trajectory.times / 2.0))
    np.testing.assert_allclose(trajectory.get_values('belief.x'), expected, rtol=1e-7, atol=1e-9)


def test_a_goal_of_a_number_pulls_every_component_of_a_vector():
    """Test that a goal given as a number pulls every component of a vector state towards it."""
    system = System(
        elements=[Element('point', values={'x': np.array([0.0, 2.0])})],
        interactions=[RelaxationTimeInteraction('relaxation', target='point', goal=1.0, time=0.1)],
    )
    assert system.prepare().simulate(until=5.0, dt=5.0).get_values('point.x')[-1] == pytest.approx([1.0, 1.0])


def test_prepare_rejects_a_goal_that_is_not_a_function():
    """Test that `prepare` rejects a goal that is neither callable nor a number."""
    with pytest.raises(ValidationError, match='goal'):
        RelaxationTimeInteraction('ou', target='walkers', goal='zero').prepare()
