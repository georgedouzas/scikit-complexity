"""Test the `_particles` module of `skcomplex.core.engine.compile.passes`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw, UpdateLaw
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.systems import System


def _step(agent, system):
    """Move an agent one unit along `x`."""
    del system
    agent.x += 1.0


def _get_walkers(boundary, order):
    """Return two walkers in a box of side 10 whose `x` and `y` follow rate laws of an order."""
    box = ContinuousSpace(extent=(10.0, 10.0), boundary=boundary)
    walkers = Element('walkers', values={'x': [9.0, 1.0], 'y': [5.0, 5.0]}, size=2, space=box)

    def laws(s):
        if order == 1:
            return [RateLaw(s.target.x.diff(s.t), 1.0), RateLaw(s.target.y.diff(s.t), -2.0)]
        return [RateLaw(s.target.x.diff(s.t, 2), 0.0), RateLaw(s.target.y.diff(s.t, 2), 0.0)]

    drift = Interaction('drift', roles={'target': 'walkers'}, laws=laws)
    return System(elements=[walkers], interactions=[drift], space=box).prepare()


def test_periodic_boundary_wraps_scalar_positions_named_after_the_axes():
    """Test that walkers crossing a periodic wall come back on the opposite side, along `x` and along `y`."""
    trajectory = _get_walkers('periodic', 1).simulate(until=4.0, dt=1.0)
    assert trajectory.get_values('walkers.x')[-1].tolist() == pytest.approx([3.0, 5.0])
    assert trajectory.get_values('walkers.y')[-1].tolist() == pytest.approx([7.0, 7.0])


def test_reflecting_boundary_mirrors_scalar_positions_and_reverses_their_velocity():
    """Test that a walker moving at speed 2 bounces off the wall at 10 and comes back."""
    trajectory = _get_walkers('reflecting', 2).simulate(
        until=2.0,
        dt=1.0,
        initial_state={"walkers.x'": [2.0, 2.0]},
    )
    np.testing.assert_allclose(trajectory.get_values('walkers.x'), [[9.0, 1.0], [9.0, 3.0], [7.0, 5.0]])
    assert trajectory.get_values("walkers.x'")[-1].tolist() == pytest.approx([-2.0, 2.0])


def test_periodic_boundary_leaves_agents_to_their_space():
    """Test that agents whose rule walks `x` past a periodic wall run, with no boundary law on their positions."""
    box = ContinuousSpace(extent=(10.0, 10.0), boundary='periodic')
    walkers = Element('walkers', values={'x': [8.5, 1.0], 'y': [5.0, 5.0]}, size=2, space=box)
    walk = Interaction('walk', roles={'target': 'walkers'}, laws=lambda s: [UpdateLaw(s.target, _step)])
    trajectory = System(elements=[walkers], interactions=[walk], space=box).prepare().simulate(until=2)
    np.testing.assert_allclose(trajectory.get_agents('walkers', 2).locations, [[0.5, 5.0], [3.0, 5.0]])
