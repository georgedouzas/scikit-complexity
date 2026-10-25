"""Test the `_spin_glasses` module of `skcomplex.models.physics.statistical_mechanics`."""

import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.models import describe
from skcomplex.models.physics.statistical_mechanics import (
    edwards_anderson,
    mattis_spin_glass,
)

ISING_CRITICAL_TEMPERATURE = 2 / np.log(1 + np.sqrt(2))


def onsager_magnetization(temperature):
    """Return the spontaneous magnetization of the square-lattice Ising model below its critical temperature."""
    return (1 - np.sinh(2 / temperature) ** -4) ** (1 / 8)


def lattice_bond_energy(spins, right, down):
    """Return the energy per spin `-sum_<ij> J_ij s_i s_j / N` of configurations on a periodic square lattice."""
    bonds = spins * right * np.roll(spins, -1, axis=-1) + spins * down * np.roll(spins, -1, axis=-2)
    return -bonds.sum(axis=(-2, -1)) / (spins.shape[-1] * spins.shape[-2])


def test_edwards_anderson_freezes_without_magnetization():
    """Test that the bimodal glass freezes near its ground-state energy at low temperature, with no magnetization."""
    for temperature in (3.0, 0.5):
        system = edwards_anderson(32, temperature=temperature, random_state=0)
        values = system.elements[0].values
        right, down = (values[name].reshape(32, 32) for name in ('J_right', 'J_down'))
        trajectory = (
            Simulation(system, dt=10, random_state=0, observables=describe('edwards_anderson').observables_of(system))
            .run(until=2000)
            .trajectory_
        )
        spins = trajectory.lattice('spins.spin')[-50:]
        energy = lattice_bond_energy(spins, right, down).mean()
        frozen = np.mean(spins.mean(axis=0) ** 2)
        assert np.abs(trajectory.values('magnetization')[-50:]).mean() < 0.1
        if temperature > 1:
            assert abs(energy + 2 * np.tanh(1 / temperature)) < 0.05
            assert frozen < 0.1
        else:
            assert abs(energy + 1.40) < 0.03
            assert frozen > 0.6


def test_mattis_spin_glass_overlap_orders_like_ising():
    """Test that the Mattis overlap follows the Onsager magnetization below `T_c` while the magnetization stays zero."""
    for temperature in (1.5, 3.5):
        system = mattis_spin_glass(32, temperature=temperature, random_state=0)
        trajectory = (
            Simulation(system, dt=10, random_state=0, observables=describe('mattis_spin_glass').observables_of(system))
            .run(until=1000)
            .trajectory_
        )
        overlap = np.abs(trajectory.values('overlap')[-30:]).mean()
        assert np.abs(trajectory.values('magnetization')[-30:]).mean() < 0.1
        if temperature < ISING_CRITICAL_TEMPERATURE:
            assert abs(overlap - onsager_magnetization(temperature)) < 0.02
        else:
            assert overlap < 0.15
