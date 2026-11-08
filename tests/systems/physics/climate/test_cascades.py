"""Test the leading-following cascade of the `_tipping_types` module of `skcomplex.systems.physics.climate`."""

import math

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.errors import ValidationError
from skcomplex.systems.physics.climate import LeadingFollowingCascadeSystem


def _final(system):
    """Return the trajectory of a cascade run with a fixed seed to time 500."""
    return simulate(system, until=500.0, dt=0.5, solver=EulerMaruyamaSolver(dt=0.01), random_state=0)


@pytest.mark.parametrize(('coupling', 'tips'), [(0.4, True), (0.2, False)])
def test_fold_follower_tips_only_when_the_tipped_leader_pushes_it_past_its_fold(coupling, tips):
    """Test that the follower tips when `coupling x` of the tipped leader exceeds `2 / (3 √3)`, a domino cascade."""
    trajectory = _final(LeadingFollowingCascadeSystem(coupling=coupling).prepare().system_)
    leader, follower = (np.ravel(trajectory.values(name)) for name in ('leader.x', 'follower.y'))
    assert leader[0] < -1
    assert leader[-1] > 1
    assert (coupling * 1.32 > 2 / (3 * math.sqrt(3))) == tips
    assert (follower[-1] > 0) == tips


def test_hopf_follower_oscillates_after_the_leader_tips():
    """Test that the tipped leader carries a Hopf follower onto a cycle of radius `sqrt(coupling x)`."""
    trajectory = _final(LeadingFollowingCascadeSystem('hopf').prepare().system_)
    leader, y, z = (np.ravel(trajectory.values(name)) for name in ('leader.x', 'follower.y', 'follower.z'))
    radius = np.hypot(y, z)
    times = np.asarray(trajectory.times)
    assert radius[times < 50].max() < 0.15
    assert radius[-50:].mean() == pytest.approx(math.sqrt(0.4 * leader[-50:].mean()), abs=0.03)


def test_unknown_follower_is_refused():
    """Test that a follower other than a fold or a Hopf normal form is refused."""
    with pytest.raises(ValidationError, match='Unknown follower'):
        LeadingFollowingCascadeSystem('saddle').prepare()
