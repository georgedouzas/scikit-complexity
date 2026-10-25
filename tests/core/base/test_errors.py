"""Test the `_errors` module of `skcomplex.core.base`."""

import pytest

from skcomplex.core.base import ValidationError
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation

from .conftest import PersonComponent, ProgressionInteraction


def test_checked_mode_catches_values_outside_alphabet():
    """Test that checked mode names the state when a law leaves the alphabet."""
    system = System(
        components=[PersonComponent('ann', health='recovered')],
        interactions=[ProgressionInteraction('p', target='ann')],
    )
    with pytest.raises(ValidationError, match=r'ann\.health .*categorical'):
        Simulation(system, checked=True).run(until=1)
    unchecked = Simulation(system).run(until=1).trajectory_
    assert unchecked.states[-1, 0] == 3.0
