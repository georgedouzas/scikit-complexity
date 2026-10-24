"""Test the `_charged` module of `skcomplex.models.physics.waves`."""

import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.models.physics.waves import charged_schrodinger


def _packet_centres(**potential):
    """Return the times and the centres of the charged packet over a run."""
    trajectory = Simulation(charged_schrodinger(256, **potential), dt=0.5).run(until=6).trajectory_
    density = trajectory['packet.u'] ** 2 + trajectory['packet.w'] ** 2
    positions = 80 / 256 * np.arange(256) - 40
    return trajectory.times, density @ positions / density.sum(axis=1)


def test_charged_schrodinger_is_accelerated_by_the_field_and_not_by_a_gauge():
    """Test that the field pulse accelerates the packet at 2 D q E and a pure gauge cancels the canonical momentum."""
    times, centres = _packet_centres(E=1.0, duration=2.0)
    expected = np.where(times < 2.0, 0.5 * times**2, 2.0 + 2.0 * (times - 2.0))
    np.testing.assert_allclose(centres, expected, atol=1e-3)
    _, centres = _packet_centres(E=0.0, A_x=1.0, momentum=1.0)
    np.testing.assert_allclose(centres, 0.0, atol=1e-6)
