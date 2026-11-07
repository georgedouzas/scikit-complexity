"""Test the `_traps` module of `skcomplex.systems.physics.soft_matter`."""

import math

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.soft_matter import BiasedTrapWalkSystem


def test_biased_trap_walk_subdiffusion():
    """Test that walkers among traps of exponent 1/2 travel as t^(1/2) with the renewal prefactor."""
    alpha = 0.5
    trajectory = simulate(BiasedTrapWalkSystem(100, n_sites=200, alpha=alpha), random_state=0)
    times, position = trajectory.times, trajectory.values('position')
    late = times >= 50
    slope = np.polyfit(np.log(times[late]), np.log(position[late]), 1)[0]
    assert slope == pytest.approx(alpha, abs=0.1)
    prediction = times[-1] ** alpha * math.sin(math.pi * alpha) / (math.pi * alpha)
    assert position[-1] == pytest.approx(prediction, rel=0.2)
