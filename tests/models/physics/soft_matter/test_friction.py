"""Test the `_friction` module of `skcomplex.models.physics.soft_matter`."""

import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.models.physics.soft_matter import (
    shaken_spring_block,
)


def _elongation_statistics(positions):
    """Return the energy per spring and the correlation length of the elongations of blocked chains."""
    elongations = np.diff(positions, axis=1)
    correlations = np.array(
        [np.mean(elongations[:, : elongations.shape[1] - shift] * elongations[:, shift:]) for shift in range(30)],
    )
    correlations /= correlations[0]
    positive = correlations[: np.argmax(correlations < 0)] if np.any(correlations < 0) else correlations
    return np.mean(elongations**2 / 2), positive.sum()


def test_shaken_spring_block_blocked_configurations():
    """Test that every cycle ends blocked and that stronger shaking gives more energy and longer correlations."""
    statistics = []
    for amplitude, relax_time in ((1.0, 10.0), (3.0, 38.0)):
        period = 2.0 + relax_time
        system = shaken_spring_block(50, amplitude=amplitude, relax_time=relax_time)
        simulation = Simulation(system, dt=period, solver=EulerMaruyama(dt=0.01), random_state=0)
        trajectory = simulation.run(until=12 * period).trajectory_
        positions, velocities = trajectory['chain.x'][1:], trajectory['chain.v'][1:]
        assert np.all(np.abs(velocities) <= 0.02)
        assert np.all(np.abs(np.diff(positions, n=2, axis=1)) < 1.0)
        statistics.append(_elongation_statistics(positions))
    (weak_energy, weak_length), (strong_energy, strong_length) = statistics
    assert strong_energy > 2 * weak_energy
    assert strong_length > 2 * weak_length


def test_shaken_spring_block_independent_anchored_blocks():
    """Test that independent blocks held by springs stop inside the blocked interval |x| < mu_s / anchor."""
    system = shaken_spring_block(100, k=0.0, anchor=2.0, amplitude=2.0, relax_time=20.0)
    simulation = Simulation(system, dt=22.0, solver=EulerMaruyama(dt=0.01), random_state=0).run(until=110)
    positions = simulation.trajectory_['chain.x'][1:]
    assert np.all(np.abs(positions) < 0.5)
    assert np.abs(positions).max() > 0.3
