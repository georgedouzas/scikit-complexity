"""Test boxes, pair potentials, and ideal gases of the physics module."""

import numpy as np
import pytest

from skcomplex.core.base import Domain, parameter
from skcomplex.core.model import EuclideanSpace, System
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.models.physics import PairPotential, Particle, PeriodicBox, ideal_gas

N_DIM = 2


class SoftSphere(PairPotential):
    """Soft repulsion `k * (cutoff - r) ** 2 / 2`."""

    k: float = parameter(50.0, domain=Domain.POSITIVE)

    def pair_energy(self, s, r):
        """Return the soft repulsion energy."""
        return s.k * (s.cutoff - r) ** 2 / 2


def _wall_pressure_times_area(trajectory, size):
    """Return the pressure times area measured from wall impulses."""
    velocities = trajectory['gas.v']
    flipped = np.sign(velocities[1:]) != np.sign(velocities[:-1])
    impulse = float((2 * np.abs(velocities[1:]) * flipped).sum())
    return impulse / (2 * N_DIM * size * trajectory.times[-1]) * size**2


@pytest.mark.parametrize(('kT', 'size'), [(1.0, 1.0), (2.0, 3.0)])
def test_ideal_gas_obeys_ideal_gas_law(kT, size):
    """Test that pressure times area equals N times kT within 2 percent."""
    n = 2000
    trajectory = (
        Simulation(ideal_gas(n, kT=kT, size=size, random_state=0), dt=0.002 * size).run(until=4 * size).trajectory_
    )
    velocities = trajectory['gas.v'][0]
    measured_kT = float((velocities**2).sum(axis=1).mean()) / N_DIM
    assert _wall_pressure_times_area(trajectory, size) == pytest.approx(n * measured_kT, rel=0.02)


def test_ideal_gas_molecules_stay_inside_and_keep_speed():
    """Test that molecules stay inside the box and keep their speed."""
    trajectory = Simulation(ideal_gas(300, kT=3.0, random_state=2), dt=0.01).run(until=5).trajectory_
    positions, velocities = trajectory['gas.x'], trajectory['gas.v']
    assert np.all((positions >= 0) & (positions <= 1))
    speeds = np.linalg.norm(velocities, axis=2)
    assert np.allclose(speeds, speeds[0])


def test_ideal_gas_initial_temperature_and_momentum():
    """Test that the ideal gas starts at its temperature with zero total momentum."""
    system = ideal_gas(20_000, kT=2.0, mass=4.0, n_dim=3, random_state=3)
    velocities = system.elements[0].values['v']
    assert np.allclose(velocities.sum(axis=0), 0.0)
    assert 4.0 * float((velocities**2).mean()) == pytest.approx(2.0, rel=0.02)


def test_periodic_box_wraps_member_with_same_velocity():
    """Test that a member leaving one side reappears at the opposite side with the same velocity."""
    system = System(
        elements=[Particle.group('gas', 1, x=[[0.95, 0.5]], v=[[1.0, 0.0]])],
        interactions=[PeriodicBox('box', target='gas', size=1.0)],
        space=EuclideanSpace(2),
    )
    trajectory = Simulation(system, dt=0.1).run(until=0.2).trajectory_
    assert np.allclose(trajectory['gas.x'][-1], [[0.15, 0.5]])
    assert np.allclose(trajectory['gas.v'][-1], [[1.0, 0.0]])


def test_pair_interaction_conserves_momentum():
    """Test that soft-sphere collisions inside an isolated group keep the total momentum."""
    rng = np.random.default_rng(0)
    system = System(
        elements=[Particle.group('gas', 200, x=rng.uniform(0, 3, (200, 2)), v=rng.normal(0, 1, (200, 2)))],
        interactions=[SoftSphere('soft', first='gas', second='gas', cutoff=0.2)],
        space=EuclideanSpace(2),
    )
    trajectory = Simulation(system, solver=ScipyODE(rtol=1e-9, atol=1e-12), dt=0.05).run(until=2).trajectory_
    momentum = trajectory['gas.v'].sum(axis=1)
    assert np.allclose(momentum, momentum[0], atol=1e-8)
