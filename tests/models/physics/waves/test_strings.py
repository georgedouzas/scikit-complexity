"""Test the `_strings` module of `skcomplex.models.physics.waves`."""

import math

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models.physics.waves import spring_chain, standing_waves

from ..conftest import _run, _series


def test_plucked_spring_chain_keeps_its_fundamental_mode():
    """Test that the fundamental transverse mode of the chain oscillates at `2 sqrt(k / m) sin(pi / (2 (N + 1)))`."""
    n_masses = 15
    trajectory = Simulation(spring_chain(n_masses), dt=0.1).prepare().run(until=4.0).trajectory_
    heights = np.stack([_series(trajectory, f'mass_{index}.x')[0][:, 1] for index in range(n_masses)], axis=1)
    mode = np.sin(np.pi * np.arange(1, n_masses + 1) / (n_masses + 1))
    amplitude = heights @ mode / (mode @ mode)
    frequency = 2 * math.sqrt(1.0 / 0.01) * math.sin(math.pi / (2 * (n_masses + 1)))
    assert amplitude == pytest.approx(amplitude[0] * np.cos(frequency * trajectory.times), abs=2e-4)


def test_standing_wave_returns_after_its_period():
    """Test that a standing wave on a string with fixed ends returns to its shape after `2 L / (n c)`."""
    n_sites, length, mode = 49, 2.0, 2
    spacing = length / (n_sites + 1)
    frequency = 2 / spacing * math.sin(mode * math.pi / (2 * (n_sites + 1)))
    period = 2 * math.pi / frequency
    trajectory = _run(standing_waves(n_sites, length=length, mode=mode), period, period / 20)
    shape = _series(trajectory, 'string.u')[0]
    assert period == pytest.approx(2 * length / mode, rel=1e-3)
    assert np.allclose(shape[-1], shape[0], atol=1e-7)
    assert np.allclose(shape[10], -shape[0], atol=1e-7)
