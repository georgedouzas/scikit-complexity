"""Test the `_phase_oscillator` module of `skcomplex.systems.physics.common.oscillators`."""

import math

import pytest

from skcomplex.core.systems import System
from skcomplex.systems.physics.common.oscillators import PhaseOscillatorElement, PhaseRotationInteraction


def _turn(clock, until):
    """Return the phases of a clock turned at its natural frequency."""
    system = System(elements=[clock], interactions=[PhaseRotationInteraction('turn', target='clock')]).prepare()
    return system.simulate(until=until, dt=0.5).get_values('clock.theta')


def test_phase_turns_at_the_natural_frequency():
    """Test that a free phase grows linearly at its natural frequency."""
    assert _turn(PhaseOscillatorElement('clock', theta=0.5, omega=-1.5), 1.0)[-1] == pytest.approx(0.5 - 1.5)


def test_phase_wraps_around_the_circle():
    """Test that a phase that turns past half a turn comes back onto the circle around 0."""
    last = _turn(PhaseOscillatorElement('clock', omega=1.0), 4.0)[-1]
    assert last == pytest.approx(4.0 - 2 * math.pi)
