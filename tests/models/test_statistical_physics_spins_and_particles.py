"""Test the mean-field and disordered spin models and the driven particle models of statistical mechanics."""

import itertools

import numpy as np
import pytest
from scipy.optimize import brentq

from skcomplex.core.runtime import Simulation
from skcomplex.models import describe
from skcomplex.models.physics.statistical_mechanics import (
    absorbing_birth_death,
    asep,
    curie_weiss,
    dissipative_zero_range,
    edwards_anderson,
    mattis_spin_glass,
    sherrington_kirkpatrick,
    zero_range_process,
)

ISING_CRITICAL_TEMPERATURE = 2 / np.log(1 + np.sqrt(2))


def spontaneous_magnetization(temperature):
    """Return the positive root of `m = tanh(m / T)`, the Curie-Weiss magnetization for `J = 1`."""
    return brentq(lambda m: m - np.tanh(m / temperature), 1e-6, 1.0)


def onsager_magnetization(temperature):
    """Return the spontaneous magnetization of the square-lattice Ising model below its critical temperature."""
    return (1 - np.sinh(2 / temperature) ** -4) ** (1 / 8)


def lattice_bond_energy(spins, right, down):
    """Return the energy per spin `-sum_<ij> J_ij s_i s_j / N` of configurations on a periodic square lattice."""
    bonds = spins * right * np.roll(spins, -1, axis=-1) + spins * down * np.roll(spins, -1, axis=-2)
    return -bonds.sum(axis=(-2, -1)) / (spins.shape[-1] * spins.shape[-2])


def tree_occupation_probabilities(*, n_levels, branches, nu, alpha, beta_ext, dissipation):
    """Return the probabilities `z_j` that a node of level `j` is occupied, from the balance of currents of Bertin."""
    rates = nu * (branches ** np.arange(n_levels - 1)) ** alpha
    matrix, source = np.zeros((n_levels, n_levels)), np.zeros(n_levels)
    source[0] = nu * np.exp(-beta_ext)
    matrix[0, 0] = nu
    for level in range(n_levels - 1):
        down = branches * rates[level]
        matrix[level, level] += down
        matrix[level, level + 1] -= down
        matrix[level + 1, level + 1] += rates[level]
        matrix[level + 1, level] -= rates[level]
    matrix[-1, -1] += dissipation
    return np.linalg.solve(matrix, source)


@pytest.mark.parametrize('temperature', [0.5, 2.0])
def test_curie_weiss_magnetization(temperature):
    """Test that 2000 spins order at `T = 0.5 J` as `m = tanh(m J / T)` predicts and stay disordered at `T = 2 J`."""
    system = curie_weiss(2000, temperature=temperature, random_state=1)
    trajectory = (
        Simulation(system, dt=1.0, random_state=0, observables=describe('curie_weiss').observables_of(system))
        .run(until=30)
        .trajectory_
    )
    magnetization = np.abs(trajectory['magnetization'][10:]).mean()
    if temperature < 1:
        assert abs(magnetization - spontaneous_magnetization(temperature)) < 0.02
    else:
        assert magnetization < 0.05


def test_sherrington_kirkpatrick_energy_and_freezing():
    """Test that the energy above `T_c` is `-1 / (2 T)` and that below `T_c` the spins freeze with no magnetization."""
    frozen = []
    for temperature in (2.0, 0.3):
        system = sherrington_kirkpatrick(200, temperature=temperature, random_state=0)
        couplings = system.elements[0].values['couplings']
        trajectory = (
            Simulation(
                system,
                dt=1.0,
                random_state=0,
                observables=describe('sherrington_kirkpatrick').observables_of(system),
            )
            .run(until=100)
            .trajectory_
        )
        spins = trajectory['spins.spin'][50:]
        energy = -0.5 * np.einsum('ti,ij,tj->t', spins, couplings, spins).mean() / 200
        frozen.append(np.mean(spins.mean(axis=0) ** 2))
        if temperature > 1:
            assert abs(energy + 1 / (2 * temperature)) < 0.03
        else:
            assert energy < -0.65
    assert frozen[0] < 0.1
    assert frozen[1] > 0.6


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
        assert np.abs(trajectory['magnetization'][-50:]).mean() < 0.1
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
        overlap = np.abs(trajectory['overlap'][-30:]).mean()
        assert np.abs(trajectory['magnetization'][-30:]).mean() < 0.1
        if temperature < ISING_CRITICAL_TEMPERATURE:
            assert abs(overlap - onsager_magnetization(temperature)) < 0.02
        else:
            assert overlap < 0.15


def test_absorbing_birth_death_metastable_and_absorbed():
    """Test that the active state fluctuates around `rho_0` with the width of Bertin and that below `nu` it dies out."""
    active = Simulation(absorbing_birth_death(kappa=2.0), dt=0.5, random_state=0, backend='numba').run(until=1000)
    density = active.trajectory_['particles.n'][100:] / 100
    assert abs(density.mean() - 1.0) < 0.05
    assert abs(density.std() / density.mean() - np.sqrt(2.0 / 100)) < 0.02
    absorbed = Simulation(absorbing_birth_death(kappa=0.5), dt=0.5, random_state=0, backend='numba').run(until=100)
    assert absorbed.trajectory_['particles.n'][-1] == 0


def test_zero_range_process_condensation():
    """Test that above the critical density `1 / (b - 2)` one site holds the excess particles and below it none does."""
    for density in (3.0, 0.25):
        system = zero_range_process(50, density=density, b=4.0)
        trajectory = Simulation(system, dt=100.0, random_state=0, backend='numba').run(until=20000).trajectory_
        occupations = trajectory['sites.n'][50:]
        assert (occupations.sum(axis=1) == round(density * 50)).all()
        largest = occupations.max(axis=1).mean()
        if density > 0.5:
            assert abs(largest - 50 * (density - 0.5)) < 20
        else:
            assert largest < 8


def test_dissipative_zero_range_level_profile():
    """Test that the occupation of every level matches the balance of currents of the product stationary state."""
    settings = {'n_levels': 4, 'branches': 2, 'nu': 1.0, 'alpha': 0.5, 'beta_ext': 1.0, 'dissipation': 0.5}
    system = dissipative_zero_range(**settings)
    trajectory = Simulation(system, dt=1.0, random_state=0, backend='numba').run(until=6000).trajectory_
    quanta = trajectory['nodes.n'][100:]
    starts = [0, 1, 3, 7, 15]
    occupied = [(quanta[:, start:stop] > 0).mean() for start, stop in itertools.pairwise(starts)]
    expected = tree_occupation_probabilities(**settings)
    np.testing.assert_allclose(occupied, expected, atol=0.02)
    assert np.all(np.diff(expected) < 0)


@pytest.mark.parametrize(
    ('alpha', 'gamma', 'current', 'bulk'),
    [(0.2, 0.7, 0.16, 0.2), (0.7, 0.2, 0.16, 0.8), (0.8, 0.8, 0.25, 0.5)],
)
def test_asep_phases(alpha, gamma, current, bulk):
    """Test the current and the bulk density of the low-density, high-density, and maximal-current phases."""
    system = asep(100, alpha=alpha, gamma=gamma)
    trajectory = Simulation(system, dt=2.0, random_state=0, backend='numba').run(until=4000).trajectory_
    occupations = trajectory['lane.n'][250:]
    assert abs(gamma * occupations[:, -1].mean() - current) < 0.02
    assert abs(occupations[:, 40:60].mean() - bulk) < 0.05
