"""Test the `_gases` module of `skcomplex.models.physics.statistical_mechanics`."""

import numpy as np
import pytest

from skcomplex.engine.runtime import Simulation
from skcomplex.models.physics.statistical_mechanics import ideal_gas

N_DIM = 2


def _wall_pressure_times_area(trajectory, size):
    """Return the pressure times area measured from wall impulses."""
    velocities = trajectory.values('gas.v')
    flipped = np.sign(velocities[1:]) != np.sign(velocities[:-1])
    impulse = float((2 * np.abs(velocities[1:]) * flipped).sum())
    return impulse / (2 * N_DIM * size * trajectory.times[-1]) * size**2


@pytest.mark.parametrize(('kT', 'size'), [(1.0, 1.0), (2.0, 3.0)])
def test_ideal_gas_obeys_ideal_gas_law(kT, size):
    """Test that pressure times area equals N times kT within 2 percent."""
    n = 500
    trajectory = (
        Simulation(ideal_gas(n, kT=kT, size=size, random_state=0), dt=0.002 * size)
        .check()
        .prepare()
        .run(until=4 * size)
        .trajectory_
    )
    velocities = trajectory.values('gas.v')[0]
    measured_kT = float((velocities**2).sum(axis=1).mean()) / N_DIM
    assert _wall_pressure_times_area(trajectory, size) == pytest.approx(n * measured_kT, rel=0.02)


def test_ideal_gas_molecules_stay_inside_and_keep_speed():
    """Test that molecules stay inside the box and keep their speed."""
    trajectory = Simulation(ideal_gas(100, kT=3.0, random_state=2), dt=0.01).check().prepare().run(until=5).trajectory_
    positions, velocities = trajectory.values('gas.x'), trajectory.values('gas.v')
    assert np.all((positions >= 0) & (positions <= 1))
    speeds = np.linalg.norm(velocities, axis=2)
    assert np.allclose(speeds, speeds[0])


def test_ideal_gas_initial_temperature_and_momentum():
    """Test that the ideal gas starts at its temperature with zero total momentum."""
    system = ideal_gas(20_000, kT=2.0, mass=4.0, n_dim=3, random_state=3)
    velocities = system.elements[0].values['v']
    assert np.allclose(velocities.sum(axis=0), 0.0)
    assert 4.0 * float((velocities**2).mean()) == pytest.approx(2.0, rel=0.02)
