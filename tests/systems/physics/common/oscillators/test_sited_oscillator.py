"""Test the `_sited_oscillator` module of `skcomplex.systems.physics.common.oscillators`."""

import numpy as np
import pytest

from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.oscillators import PhaseRotationInteraction, SitedOscillatorElement


def test_sited_oscillator_keeps_its_place_and_turns():
    """Test that a sited oscillator turns at its natural frequency and stays where it is without interactions."""
    unit = SitedOscillatorElement('unit', x=[1.0, 2.0], omega=2.0, space=ContinuousSpace(n_dim=2))
    system = System(elements=[unit], interactions=[PhaseRotationInteraction('turn', target='unit')]).prepare()
    trajectory = system.simulate(until=1.0, dt=0.5)
    assert trajectory.get_values('unit.theta')[-1] == pytest.approx(2.0)
    np.testing.assert_allclose(np.ravel(trajectory.get_values('unit.x')[-1]), [1.0, 2.0])
