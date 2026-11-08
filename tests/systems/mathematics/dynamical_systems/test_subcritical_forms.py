"""Test the `_subcritical_forms` module of `skcomplex.systems.mathematics.dynamical_systems`."""

import math

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.mathematics.dynamical_systems import (
    SubcriticalHopfFormSystem,
)


def test_subcritical_hopf_form_is_bistable():
    """Test that the focus and the cycle of radius `r² = (1 + √(1 + 4 alpha)) / 2` attract small and large starts."""
    alpha = -0.1
    radius = math.sqrt((1 + math.sqrt(1 + 4 * alpha)) / 2)
    small = simulate(SubcriticalHopfFormSystem(0.1, alpha=alpha).prepare().system_, until=100, dt=0.1).states[-1]
    large = simulate(SubcriticalHopfFormSystem(1.0, alpha=alpha).prepare().system_, until=100, dt=0.1).states[-1]
    assert np.hypot(*small) < 1e-3
    assert np.hypot(*large) == pytest.approx(radius, abs=1e-4)
    gone = simulate(SubcriticalHopfFormSystem(1.0, alpha=-0.3).prepare().system_, until=100, dt=0.1).states[-1]
    assert np.hypot(*gone) < 1e-3


def test_subcritical_hopf_form_shows_hysteresis():
    """Test that the amplitude at alpha = -0.1 depends on whether alpha came from below the fold or from above 0."""

    def alpha(t):
        """Return -0.1 until time 50, 0.2 until 150, -0.1 until 250, and -0.3 afterwards."""
        changes = [(50, -0.1), (150, 0.2), (250, -0.1)]
        return next((value for end, value in changes if t < end - 0.05), -0.3)

    system = SubcriticalHopfFormSystem(0.01, alpha=-0.1).prepare().system_
    trajectory = simulate(system, until=350, dt=0.1, schedule={'form.alpha': alpha})
    before, after, fallen = (
        np.hypot(*trajectory.states[np.argmin(np.abs(trajectory.times - t))]) for t in (50, 250, 350)
    )
    assert before < 1e-3
    assert after > 0.9
    assert fallen < 1e-3
