"""Test the QuTiP solvers of quantum systems."""

from operator import attrgetter
from typing import ClassVar

import numpy as np
import pytest

from skcomplex.core.base import BaseInteraction, SolverError, parameter, role
from skcomplex.core.elements import LatticeElement
from skcomplex.core.laws import DissipatorLaw, GeneratorLaw
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import LindbladSolver, QuantumTrajectoriesSolver, UnitarySolver
from skcomplex.models.physics.realizations.quantum import QubitElement, SpinElement

OMEGA = 2.0
GAMMA = 0.5


class RabiElement(QubitElement):
    """Rabi element."""

    omega = parameter(OMEGA)

    def laws(self, s):
        """Return the laws of the rabi element."""
        return [GeneratorLaw(s.omega / 2 * s.x)]


class DecayingElement(QubitElement):
    """Decaying element."""

    gamma = parameter(GAMMA)

    def laws(self, s):
        """Return the laws of the decaying element."""
        return [DissipatorLaw(s.minus, s.gamma)]


class DephasedElement(QubitElement):
    """Dephased element."""

    def laws(self, s):
        """Return the laws of the dephased element."""
        return [GeneratorLaw(s.x), DissipatorLaw(s.z, 0.3)]


class XXZInteraction(BaseInteraction):
    """Anisotropic exchange."""

    over: ClassVar[str] = 'neighbors'
    target: str = role()

    delta: float = parameter(1.0)

    def laws(self, s):
        """Return the exchange of one bond."""
        a, b = s.first, s.second
        return [GeneratorLaw(a.x * b.x + a.y * b.y + s.delta * a.z * b.z)]


def total_z(s):
    """Return the total magnetisation of the chain."""
    return sum((member.z for member in s.chain), start=0 * s.chain[0].z)


Z = [expectation('z', attrgetter('q.z'))]
MAGNETIZATION = [expectation('magnetization', total_z)]


def _chain(n_sites, **values):
    """Return a Heisenberg chain from a Néel state."""
    labels = ['up', 'down'] * (n_sites // 2)
    return System(
        elements=[
            LatticeElement('chain', element=SpinElement, space=LatticeSpace(n_sites), values={'initial': labels}),
        ],
        interactions=[XXZInteraction('xxz', target='chain', **values)],
    )


def test_unitary_evolution_follows_rabi_oscillations():
    """Test that a driven qubit oscillates as cos(omega t) with a constant norm."""
    system = System(elements=[RabiElement('q')])
    simulation = Simulation(system, solver='unitary', dt=0.05, observables=Z).run(until=np.pi / OMEGA)
    trajectory = simulation.trajectory_
    assert np.allclose(trajectory.values('z'), np.cos(OMEGA * trajectory.times), atol=1e-6)
    assert np.allclose(np.linalg.norm(trajectory.values('system.state'), axis=1), 1, atol=1e-8)


def test_unitary_evolution_conserves_magnetization_and_energy():
    """Test that an isotropic chain of 10 spins conserves its magnetisation and energy."""
    trajectory = (
        Simulation(_chain(10), observables=MAGNETIZATION, solver=UnitarySolver(), dt=1.0).run(until=10.0).trajectory_
    )
    assert np.ptp(trajectory.values('magnetization')) < 1e-8
    assert np.ptp(trajectory.values('energy')) < 1e-8


def test_unitary_evolution_rejects_open_systems():
    """Test that unitary evolution refuses dissipators."""
    with pytest.raises(SolverError, match='dissipators'):
        Simulation(System(elements=[DecayingElement('q')]), solver='unitary').run(until=1.0)


def test_lindblad_evolution_follows_spontaneous_decay():
    """Test that the excited population decays as exp(-gamma t) with trace 1."""
    system = System(elements=[DecayingElement('q')])
    simulation = Simulation(system, dt=0.5, observables=Z).run(until=4.0)
    assert isinstance(simulation.stepper_, LindbladSolver)
    trajectory = simulation.trajectory_
    assert np.allclose((1 + trajectory.values('z')) / 2, np.exp(-GAMMA * trajectory.times), atol=1e-6)
    traces = np.trace(trajectory.values('system.state'), axis1=1, axis2=2)
    assert np.allclose(traces, 1, atol=1e-10)


def test_lindblad_evolution_lowers_purity():
    """Test that dephasing lowers the purity from 1 frame after frame."""
    trajectory = Simulation(System(elements=[DephasedElement('q')]), dt=0.2).run(until=4.0).trajectory_
    purity = trajectory.values('purity')
    assert purity[0] == pytest.approx(1)
    assert np.all(np.diff(purity) <= 1e-10)


def test_quantum_trajectories_are_reproducible():
    """Test that two seeded trajectories of a decaying qubit are identical."""
    system = System(elements=[DecayingElement('q')])
    runs = [
        Simulation(system, solver=QuantumTrajectoriesSolver(), dt=0.5, random_state=3, observables=Z).run(until=10.0)
        for _ in range(2)
    ]
    assert np.array_equal(runs[0].trajectory_.values('z'), runs[1].trajectory_.values('z'))
    assert set(np.round(runs[0].trajectory_.values('z'), 8)) <= {-1.0, 1.0}


def test_quantum_trajectories_average_to_the_master_equation():
    """Test that 500 trajectories average to the Lindblad decay within three standard errors."""
    system = System(elements=[DecayingElement('q')])
    problem = compile_system(system, observables=Z)
    rng = np.random.default_rng(0)
    values = np.array(
        [
            Simulation(problem, solver='quantum_trajectories', dt=0.5, random_state=rng)
            .run(until=4.0)
            .trajectory_.values('z')
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
    energy = trajectory.values('energy')
    assert np.ptp(energy[:3]) < 1e-8
    assert abs(energy[3] - energy[2]) > 0.1
    assert np.ptp(energy[3:]) < 1e-8


def test_quantum_solvers_reject_classical_problems():
    """Test that a quantum solver refuses a classical problem."""
    with pytest.raises(SolverError):
        LindbladSolver().prepare(compile_system(System()), np.random.default_rng(0))
