"""Test the QuTiP solvers of quantum systems."""

from operator import attrgetter

import numpy as np
import pytest

from skcomplex.core.base import SolverError, parameter
from skcomplex.core.compile import compile_system
from skcomplex.core.model import (
    Dissipator,
    Exchange,
    Generator,
    LatticeSpace,
    QuantumSystem,
    Qubit,
    Spin,
    System,
    expectation,
)
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import LindbladEvolution, QuantumTrajectories, UnitaryEvolution

OMEGA = 2.0
GAMMA = 0.5

Rabi = Qubit.define('Rabi', parameters={'omega': OMEGA}, laws=lambda s: [Generator(s.omega / 2 * s.x)])
Decaying = Qubit.define('Decaying', parameters={'gamma': GAMMA}, laws=lambda s: [Dissipator(s.minus, s.gamma)])
Dephased = Qubit.define(
    'Dephased',
    laws=lambda s: [Generator(s.x), Dissipator(s.z, 0.3)],
)


class XXZ(Exchange):
    """Anisotropic exchange."""

    delta: float = parameter(1.0)

    def laws(self, s):
        """Return the exchange of one bond."""
        a, b = s.first, s.second
        return [Generator(a.x * b.x + a.y * b.y + s.delta * a.z * b.z)]


def total_z(s):
    """Return the total magnetisation of the chain."""
    return sum((member.z for member in s.chain), start=0 * s.chain[0].z)


Z = [expectation('z', attrgetter('q.z'))]
MAGNETIZATION = [expectation('magnetization', total_z)]


def _chain(n_sites, **values):
    """Return a Heisenberg chain from a Néel state."""
    labels = ['up', 'down'] * (n_sites // 2)
    return QuantumSystem(
        components=[Spin.lattice('chain', LatticeSpace(n_sites), initial=labels)],
        interactions=[XXZ('xxz', target='chain', **values)],
    )


def test_unitary_evolution_follows_rabi_oscillations():
    """Test that a driven qubit oscillates as cos(omega t) with a constant norm."""
    system = System(components=[Rabi('q')])
    simulation = Simulation(system, solver='unitary', dt=0.05, observables=Z).run(until=np.pi / OMEGA)
    trajectory = simulation.trajectory_
    assert np.allclose(trajectory['z'], np.cos(OMEGA * trajectory.times), atol=1e-6)
    assert np.allclose(np.linalg.norm(trajectory['system.state'], axis=1), 1, atol=1e-8)


def test_unitary_evolution_conserves_magnetization_and_energy():
    """Test that an isotropic chain of 10 spins conserves its magnetisation and energy."""
    trajectory = (
        Simulation(_chain(10), observables=MAGNETIZATION, solver=UnitaryEvolution(), dt=1.0).run(until=10.0).trajectory_
    )
    assert np.ptp(trajectory['magnetization']) < 1e-8
    assert np.ptp(trajectory['energy']) < 1e-8


def test_unitary_evolution_rejects_open_systems():
    """Test that unitary evolution refuses dissipators."""
    with pytest.raises(SolverError, match='dissipators'):
        Simulation(System(components=[Decaying('q')]), solver='unitary').run(until=1.0)


def test_lindblad_evolution_follows_spontaneous_decay():
    """Test that the excited population decays as exp(-gamma t) with trace 1."""
    system = System(components=[Decaying('q')])
    simulation = Simulation(system, dt=0.5, observables=Z).run(until=4.0)
    assert isinstance(simulation.stepper_, LindbladEvolution)
    trajectory = simulation.trajectory_
    assert np.allclose((1 + trajectory['z']) / 2, np.exp(-GAMMA * trajectory.times), atol=1e-6)
    traces = np.trace(trajectory['system.state'], axis1=1, axis2=2)
    assert np.allclose(traces, 1, atol=1e-10)


def test_lindblad_evolution_lowers_purity():
    """Test that dephasing lowers the purity from 1 frame after frame."""
    trajectory = Simulation(System(components=[Dephased('q')]), dt=0.2).run(until=4.0).trajectory_
    purity = trajectory['purity']
    assert purity[0] == pytest.approx(1)
    assert np.all(np.diff(purity) <= 1e-10)


def test_quantum_trajectories_are_reproducible():
    """Test that two seeded trajectories of a decaying qubit are identical."""
    system = System(components=[Decaying('q')])
    runs = [
        Simulation(system, solver=QuantumTrajectories(), dt=0.5, random_state=3, observables=Z).run(until=10.0)
        for _ in range(2)
    ]
    assert np.array_equal(runs[0].trajectory_['z'], runs[1].trajectory_['z'])
    assert set(np.round(runs[0].trajectory_['z'], 8)) <= {-1.0, 1.0}


def test_quantum_trajectories_average_to_the_master_equation():
    """Test that 500 trajectories average to the Lindblad decay within three standard errors."""
    system = QuantumSystem(components=[Decaying('q')])
    problem = compile_system(system, observables=Z)
    rng = np.random.default_rng(0)
    values = np.array(
        [
            Simulation(problem, solver='quantum_trajectories', dt=0.5, random_state=rng).run(until=4.0).trajectory_['z']
            for _ in range(500)
        ],
    )
    excited = (1 + values) / 2
    times = np.arange(9) * 0.5
    errors = excited.std(axis=0) / np.sqrt(len(excited)) + 1e-12
    assert np.all(np.abs(excited.mean(axis=0) - np.exp(-GAMMA * times)) <= 3 * errors + 1e-9)


def test_quantum_solvers_follow_parameter_updates():
    """Test that a quench of the anisotropy changes the energy after the update."""
    simulation = Simulation(_chain(6), observables=MAGNETIZATION, dt=0.5).run(until=1.0)
    simulation.update_parameters({'xxz.delta': 0.0})
    trajectory = simulation.extend(until=2.0).trajectory_
    energy = trajectory['energy']
    assert np.ptp(energy[:3]) < 1e-8
    assert abs(energy[3] - energy[2]) > 0.1
    assert np.ptp(energy[3:]) < 1e-8


def test_quantum_solvers_reject_classical_problems():
    """Test that a quantum solver refuses a classical problem."""
    with pytest.raises(SolverError):
        LindbladEvolution().initialize(compile_system(System()), np.random.default_rng(0))
