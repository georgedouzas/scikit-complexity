"""Test the analysis of quantum systems: spectra, ground states, purity, and entanglement."""

import numpy as np
import pytest

from skcomplex.core.analysis import (
    entanglement_entropy,
    equations_of,
    ground_state,
    hamiltonian_of,
    purity,
    spectrum,
)
from skcomplex.core.base import AnalysisError, ParameterError
from skcomplex.core.compile import compile_system
from skcomplex.core.runtime import Simulation
from skcomplex.core.system import System
from skcomplex.models import create, describe
from skcomplex.models.biology.ecology import Population
from skcomplex.models.physics.realizations import Spin
from skcomplex.models.physics.statistical_mechanics import total_magnetization


def test_spectrum_of_the_heisenberg_ring_of_four_spins():
    """Test that the antiferromagnetic ring of four spins has the ground energy -2."""
    assert spectrum(create('quantum_heisenberg', n_sites=4), k=1).energies[0] == pytest.approx(-2.0, abs=1e-12)


def test_spectrum_uses_parameters_and_sectors():
    """Test that spectra take parameter values and a conserved sector, also with the sparse eigensolver."""
    system = create('quantum_heisenberg', n_sites=4)
    ferro = spectrum(system, k=1, parameters={'exchange.J': -1.0})
    assert ferro.energies[0] == pytest.approx(-1.0)
    sector = spectrum(system, k=1, sector=(total_magnetization, 2.0))
    assert sector.energies[0] == pytest.approx(1.0)
    large = spectrum(create('quantum_heisenberg', n_sites=12), k=1, sector=(total_magnetization, 0.0))
    assert large.energies[0] / 12 == pytest.approx(0.25 - np.log(2), abs=0.02)


def test_spectrum_rejects_bad_requests():
    """Test that unknown parameters, empty sectors, and classical systems raise errors."""
    system = create('quantum_heisenberg', n_sites=4)
    with pytest.raises(ParameterError):
        spectrum(system, parameters={'exchange.K': 1.0})
    with pytest.raises(AnalysisError, match='eigenvalue'):
        spectrum(system, sector=(total_magnetization, 7.0))
    with pytest.raises(AnalysisError, match='quantum elements'):
        spectrum(System(components=[Population('rabbits')]))


def test_ground_state_is_stationary():
    """Test that a run from the ground state keeps every expectation constant."""
    system = create('quantum_heisenberg', n_sites=6, initial='ground')
    energy, state = ground_state(system)
    assert state.shape == (64,)
    entry = describe('quantum_heisenberg')
    initial = entry.initial_state_of(n_sites=6, initial='ground')
    assert initial == {'quantum_heisenberg.state': 'ground'}
    simulation = Simulation(system, dt=0.5, observables=entry.observables_of(system), initial_state=initial)
    trajectory = simulation.run(until=3.0).trajectory_
    assert np.ptp(trajectory['energy']) < 1e-8
    assert np.ptp(trajectory['sz'], axis=0).max() < 1e-8
    assert trajectory['energy'][0] == pytest.approx(energy)


def test_entanglement_entropy_of_a_singlet_is_log_two():
    """Test that one spin of the singlet has the entropy ln 2, and the pair has none."""
    singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
    system = System(components=[Spin.group('pair', 2)])
    trajectory = Simulation(system, dt=1.0, initial_state={'system.state': singlet}).run(until=1.0).trajectory_
    assert np.allclose(entanglement_entropy(trajectory, ['pair[0]']), np.log(2), atol=1e-12)
    assert np.allclose(entanglement_entropy(trajectory, ['pair']), 0, atol=1e-10)
    with pytest.raises(AnalysisError, match='not a subsystem'):
        entanglement_entropy(trajectory, ['other'])


def test_purity_of_pure_and_dephased_runs():
    """Test that purity is 1 for closed runs and falls under dephasing."""
    closed = Simulation(create('quantum_heisenberg', n_sites=4), dt=0.5).run(until=1.0).trajectory_
    assert np.allclose(purity(closed), 1)
    opened = Simulation(create('quantum_heisenberg', n_sites=4, gamma=0.5), dt=0.5).run(until=2.0).trajectory_
    assert np.all(np.diff(purity(opened)) <= 1e-10)
    assert purity(opened)[-1] < 1
    rabbits = Simulation(System(components=[Population('rabbits')]), dt=1.0).run(until=1.0).trajectory_
    with pytest.raises(AnalysisError, match='not the trajectory'):
        purity(rabbits)


def test_hamiltonian_of_prints_operators_and_equations_of_refuses():
    """Test that a quantum system prints its Hamiltonian and has no state equations."""
    system = create('quantum_heisenberg', n_sites=2, boundary='open')
    assert repr(hamiltonian_of(compile_system(system))).startswith('J_exchange*chain[0].x*chain[1].x')
    with pytest.raises(AnalysisError, match='hamiltonian_of'):
        equations_of(system)
