"""Test the `_spin_glasses` module of `skcomplex.systems.physics.statistical_mechanics`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.statistical_mechanics import EdwardsAndersonSystem, MattisSpinGlassSystem

ISING_CRITICAL_TEMPERATURE = 2 / np.log(1 + np.sqrt(2))


def onsager_magnetization(temperature):
    """Return the spontaneous magnetization of the square-lattice Ising system below its critical temperature."""
    return (1 - np.sinh(2 / temperature) ** -4) ** (1 / 8)


def lattice_bond_energy(spins, right, down):
    """Return the energy per spin `-sum_<ij> J_ij s_i s_j / N` of configurations on a periodic square lattice."""
    bonds = spins * right * np.roll(spins, -1, axis=-1) + spins * down * np.roll(spins, -1, axis=-2)
    return -bonds.sum(axis=(-2, -1)) / (spins.shape[-1] * spins.shape[-2])


def test_edwards_anderson_freezes_without_magnetization():
    """Test that the bimodal glass freezes near its ground-state energy at low temperature, with no magnetization."""
    for temperature in (3.0, 0.5):
        system = EdwardsAndersonSystem(32, temperature=temperature, random_state=0).prepare().system_
        values = system.elements[0].values
        right, down = (values[name].reshape(32, 32) for name in ('J_right', 'J_down'))
        trajectory = simulate(system, until=2000, dt=10, random_state=0, observables=EdwardsAndersonSystem.observables)
        spins = trajectory.get_lattice('spins.spin')[-50:]
        energy = lattice_bond_energy(spins, right, down).mean()
        frozen = np.mean(spins.mean(axis=0) ** 2)
        assert np.abs(trajectory.get_values('magnetization')[-50:]).mean() < 0.1
        if temperature > 1:
            assert abs(energy + 2 * np.tanh(1 / temperature)) < 0.05
            assert frozen < 0.1
        else:
            assert abs(energy + 1.40) < 0.03
            assert frozen > 0.6


def test_mattis_spin_glass_overlap_orders_like_ising():
    """Test that the Mattis overlap follows the Onsager magnetization below `T_c` while the magnetization stays zero."""
    for temperature in (1.5, 3.5):
        system = MattisSpinGlassSystem(32, temperature=temperature, random_state=0).prepare().system_
        trajectory = simulate(system, until=1000, dt=10, random_state=0, observables=MattisSpinGlassSystem.observables)
        overlap = np.abs(trajectory.get_values('overlap')[-30:]).mean()
        assert np.abs(trajectory.get_values('magnetization')[-30:]).mean() < 0.1
        if temperature < ISING_CRITICAL_TEMPERATURE:
            assert abs(overlap - onsager_magnetization(temperature)) < 0.02
        else:
            assert overlap < 0.15
