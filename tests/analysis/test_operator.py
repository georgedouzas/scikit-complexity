"""Test the analysis of operator systems: spectra, ground states, purity, and entanglement."""

import numpy as np
import pytest

from skcomplex.analysis import OperatorAnalysis
from skcomplex.core.elements import GroupElement
from skcomplex.core.system import System
from skcomplex.engine.compile import OperatorCompiler
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import AnalysisError, ParameterError, ValidationError
from skcomplex.models import create, describe
from skcomplex.models.biology.common.populations import PopulationElement
from skcomplex.models.physics.common.quantum import SpinElement
from skcomplex.models.physics.statistical_mechanics import total_magnetization
from skcomplex.specs import SectorSpec


def _operator(kind, source, **settings):
    """Return the operator analysis of a source for one kind, run."""
    return OperatorAnalysis(kind, **settings).prepare(source).run()


def test_spectrum_of_the_heisenberg_ring_of_four_spins():
    """Test that the antiferromagnetic ring of four spins has the ground energy -2."""
    assert _operator('spectrum', create('quantum_heisenberg', n_sites=4), n_states=1).energies_[0] == pytest.approx(
        -2.0,
        abs=1e-12,
    )


def test_spectrum_uses_parameters_and_sectors():
    """Test that spectra take parameter values and a conserved sector, also with the sparse eigensolver."""
    system = create('quantum_heisenberg', n_sites=4)
    ferro = _operator('spectrum', system, n_states=1, parameters={'exchange.J': -1.0})
    assert ferro.energies_[0] == pytest.approx(-1.0)
    sector = _operator('spectrum', system, n_states=1, sector=SectorSpec(total_magnetization, 2.0))
    assert sector.energies_[0] == pytest.approx(1.0)
    large = _operator(
        'spectrum',
        create('quantum_heisenberg', n_sites=12),
        n_states=1,
        sector=SectorSpec(total_magnetization, 0.0),
    )
    assert large.energies_[0] / 12 == pytest.approx(0.25 - np.log(2), abs=0.02)


def test_spectrum_rejects_bad_requests():
    """Test that unknown parameters, empty sectors, and classical systems raise errors."""
    system = create('quantum_heisenberg', n_sites=4)
    with pytest.raises(ParameterError):
        _operator('spectrum', system, parameters={'exchange.K': 1.0})
    with pytest.raises(AnalysisError, match='eigenvalue'):
        _operator('spectrum', system, sector=SectorSpec(total_magnetization, 7.0))
    with pytest.raises(ValidationError, match='no operator element'):
        _operator('spectrum', System(elements=[PopulationElement('rabbits')]))


def test_ground_state_is_stationary():
    """Test that a run from the ground state keeps every expectation constant."""
    system = create('quantum_heisenberg', n_sites=6, initial='ground')
    ground = _operator('ground-state', system)
    energy, state = ground.energy_, ground.state_
    assert state.shape == (64,)
    entry = describe('quantum_heisenberg')
    initial = entry.initial_state_of(n_sites=6, initial='ground')
    assert initial == {'quantum_heisenberg.state': 'ground'}
    simulation = Simulation(
        system,
        dt=0.5,
        observables=entry.observables_of(system),
        initial_state=initial,
        compiler=OperatorCompiler(),
    ).prepare()
    trajectory = simulation.run(until=3.0).trajectory_
    assert np.ptp(trajectory.values('energy')) < 1e-8
    assert np.ptp(trajectory.values('sz'), axis=0).max() < 1e-8
    assert trajectory.values('energy')[0] == pytest.approx(energy)


def test_entanglement_entropy_of_a_singlet_is_log_two():
    """Test that one spin of the singlet has the entropy ln 2, and the pair has none."""
    singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
    system = System(elements=[GroupElement('pair', element=SpinElement, size=2)])
    trajectory = (
        Simulation(system, dt=1.0, initial_state={'system.state': singlet}, compiler=OperatorCompiler())
        .prepare()
        .run(until=1.0)
        .trajectory_
    )
    assert np.allclose(_operator('entanglement', trajectory, subsystems=['pair[0]']).entropy_, np.log(2), atol=1e-12)
    assert np.allclose(_operator('entanglement', trajectory, subsystems=['pair']).entropy_, 0, atol=1e-10)
    with pytest.raises(AnalysisError, match='not a subsystem'):
        _operator('entanglement', trajectory, subsystems=['other'])


def test_purity_of_pure_and_dephased_runs():
    """Test that purity is 1 for closed runs and falls under dephasing."""
    closed = (
        Simulation(create('quantum_heisenberg', n_sites=4), dt=0.5, compiler=OperatorCompiler())
        .prepare()
        .run(until=1.0)
        .trajectory_
    )
    assert np.allclose(_operator('purity', closed).purity_, 1)
    opened = (
        Simulation(create('quantum_heisenberg', n_sites=4, gamma=0.5), dt=0.5, compiler=OperatorCompiler())
        .prepare()
        .run(until=2.0)
        .trajectory_
    )
    assert np.all(np.diff(_operator('purity', opened).purity_) <= 1e-10)
    assert _operator('purity', opened).purity_[-1] < 1
    rabbits = Simulation(System(elements=[PopulationElement('rabbits')]), dt=1.0).prepare().run(until=1.0).trajectory_
    with pytest.raises(AnalysisError, match='not the trajectory'):
        _operator('purity', rabbits)


def test_settings_the_kind_ignores_are_rejected():
    """Test that the number of states of the purity, which it does not read, is rejected at check."""
    with pytest.raises(AnalysisError, match='does not read n_states'):
        OperatorAnalysis('purity', n_states=2).prepare()
