"""Test the `_phase_rotation` module of `skcomplex.systems.physics.common.oscillators`."""

import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.oscillators import PhaseRotationInteraction


def test_phase_turns_at_the_natural_frequency():
    """Test that a free phase grows linearly at its natural frequency."""
    clock = Element('clock', values={'theta': 0.5, 'omega': -1.5})
    system = System(elements=[clock], interactions=[PhaseRotationInteraction('rotation', target='clock')])
    trajectory = system.prepare().simulate(until=2.0, dt=0.5)
    assert trajectory.get_values('clock.theta')[-1] == pytest.approx(0.5 - 3.0)


def test_every_member_turns_at_its_own_frequency():
    """Test that the phases of a group turn at the frequencies of their members."""
    clocks = Element('clocks', values={'theta': [0.0, 1.0], 'omega': [2.0, -1.0]}, size=2)
    system = System(elements=[clocks], interactions=[PhaseRotationInteraction('rotation', target='clocks')])
    trajectory = system.prepare().simulate(until=1.0, dt=0.5)
    assert trajectory.get_values('clocks.theta')[-1] == pytest.approx([2.0, 0.0])
