"""Test the `_subcritical_forms` module of `skcomplex.systems.mathematics.dynamical_systems`."""

import math

import numpy as np
import pytest

from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems.mathematics.dynamical_systems import (
    SubcriticalHopfFormSystem,
)


def test_subcritical_hopf_form_is_bistable():
    """Test that the focus and the cycle of radius `r² = (1 + √(1 + 4 alpha)) / 2` attract small and large starts."""
    alpha = -0.1
    radius = math.sqrt((1 + math.sqrt(1 + 4 * alpha)) / 2)
    small = (
        Simulation(SubcriticalHopfFormSystem(0.1, alpha=alpha).prepare().system_, dt=0.1)
        .prepare()
        .run(until=100)
        .state_
    )
    large = (
        Simulation(SubcriticalHopfFormSystem(1.0, alpha=alpha).prepare().system_, dt=0.1)
        .prepare()
        .run(until=100)
        .state_
    )
    assert np.hypot(*small) < 1e-3
    assert np.hypot(*large) == pytest.approx(radius, abs=1e-4)
    gone = (
        Simulation(SubcriticalHopfFormSystem(1.0, alpha=-0.3).prepare().system_, dt=0.1).prepare().run(until=100).state_
    )
    assert np.hypot(*gone) < 1e-3


def test_subcritical_hopf_form_shows_hysteresis():
    """Test that the amplitude at alpha = -0.1 depends on whether alpha came from below the fold or from above 0."""
    simulation = Simulation(SubcriticalHopfFormSystem(0.01, alpha=-0.1).prepare().system_, dt=0.1).prepare()
    before = np.hypot(*simulation.run(until=50).state_)
    simulation.update_parameters({'form.alpha': 0.2})
    simulation.extend(until=150)
    simulation.update_parameters({'form.alpha': -0.1})
    after = np.hypot(*simulation.extend(until=250).state_)
    simulation.update_parameters({'form.alpha': -0.3})
    fallen = np.hypot(*simulation.extend(until=350).state_)
    assert before < 1e-3
    assert after > 0.9
    assert fallen < 1e-3
