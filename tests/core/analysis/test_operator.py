"""Test the analysis of operator systems: spectra, ground states, purity, and entanglement."""

import numpy as np
import pytest

from skcomplex.core.analysis import operator_measures
from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import AnalysisError, ParameterError, ValidationError
from skcomplex.core.specs import SectorSpec
from skcomplex.systems.biology.common.populations import PopulationElement
from skcomplex.systems.physics.statistical_mechanics import QuantumHeisenbergSystem, total_magnetization

from ...conftest import SpinElement


def _operator(kind, source, **settings):
    """Return the operator measures of a source for one kind."""
    return operator_measures(source, kind=kind, **settings)


def test_spectrum_of_the_heisenberg_ring_of_four_spins():
    """Test that the antiferromagnetic ring of four spins has the ground energy -2."""
    assert _operator('spectrum', QuantumHeisenbergSystem(n_sites=4).prepare().system_, n_states=1).energies[
        0
    ] == pytest.approx(
        -2.0,
        abs=1e-12,
    )


def test_spectrum_uses_parameters_and_sectors():
    """Test that spectra take parameter values and a conserved sector, also with the sparse eigensolver."""
    system = QuantumHeisenbergSystem(n_sites=4).prepare().system_
    ferro = _operator('spectrum', system, n_states=1, parameters={'exchange.J': -1.0})
    assert ferro.energies[0] == pytest.approx(-1.0)
    sector = _operator('spectrum', system, n_states=1, sector=SectorSpec(total_magnetization, 2.0))
    assert sector.energies[0] == pytest.approx(1.0)
    large = _operator(
        'spectrum',
        QuantumHeisenbergSystem(n_sites=12).prepare().system_,
        n_states=1,
        sector=SectorSpec(total_magnetization, 0.0),
    )
    assert large.energies[0] / 12 == pytest.approx(0.25 - np.log(2), abs=0.02)


def test_spectrum_rejects_bad_requests():
    """Test that unknown parameters, empty sectors, and classical systems raise errors."""
    system = QuantumHeisenbergSystem(n_sites=4).prepare().system_
    with pytest.raises(ParameterError):
        _operator('spectrum', system, parameters={'exchange.K': 1.0})
    with pytest.raises(AnalysisError, match='eigenvalue'):
        _operator('spectrum', system, sector=SectorSpec(total_magnetization, 7.0))
    with pytest.raises(ValidationError, match='no operator element'):
        _operator('spectrum', CompositeSystem(elements=[PopulationElement('rabbits')]))


def test_ground_state_is_stationary():
    """Test that a run from the ground state keeps every expectation constant."""
    system = QuantumHeisenbergSystem(n_sites=6, initial='ground').prepare().system_
    ground = _operator('ground-state', system)
    energy, state = ground.energy, ground.state
    assert state.shape == (64,)
    settings = QuantumHeisenbergSystem(n_sites=6, initial='ground').get_run_settings()
    initial = settings['initial_state']
    assert initial == {'system.state': 'ground'}
    trajectory = simulate(system, until=3.0, dt=0.5, observables=settings['observables'], initial_state=initial)
    assert np.ptp(trajectory.get_values('energy')) < 1e-8
    assert np.ptp(trajectory.get_values('sz'), axis=0).max() < 1e-8
    assert trajectory.get_values('energy')[0] == pytest.approx(energy)


def test_entanglement_entropy_of_a_singlet_is_log_two():
    """Test that one spin of the singlet has the entropy ln 2, and the pair has none."""
    singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
    system = CompositeSystem(elements=[GroupElement('pair', element=SpinElement, size=2)])
    trajectory = simulate(system, until=1.0, dt=1.0, initial_state={'system.state': singlet})
    assert np.allclose(_operator('entanglement', trajectory, subsystems=['pair[0]']).entropy, np.log(2), atol=1e-12)
    assert np.allclose(_operator('entanglement', trajectory, subsystems=['pair']).entropy, 0, atol=1e-10)
    with pytest.raises(AnalysisError, match='not a subsystem'):
        _operator('entanglement', trajectory, subsystems=['other'])


def test_purity_of_pure_and_dephased_runs():
    """Test that purity is 1 for closed runs and falls under dephasing."""
    closed = simulate(QuantumHeisenbergSystem(n_sites=4).prepare().system_, until=1.0, dt=0.5)
    assert np.allclose(_operator('purity', closed).purity, 1)
    opened = simulate(QuantumHeisenbergSystem(n_sites=4, gamma=0.5).prepare().system_, until=2.0, dt=0.5)
    assert np.all(np.diff(_operator('purity', opened).purity) <= 1e-10)
    assert _operator('purity', opened).purity[-1] < 1
    rabbits = simulate(CompositeSystem(elements=[PopulationElement('rabbits')]), until=1.0, dt=1.0)
    with pytest.raises(AnalysisError, match='not the trajectory'):
        _operator('purity', rabbits)


def test_settings_the_kind_ignores_are_rejected():
    """Test that the number of states of the purity, which it does not read, is rejected."""
    with pytest.raises(AnalysisError, match='does not read n_states'):
        operator_measures(QuantumHeisenbergSystem(n_sites=2).prepare().system_, kind='purity', n_states=2)
