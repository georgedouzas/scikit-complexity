"""Test the `UnitarySolver` class."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import CC

from skcomplex.core.elements import LatticeElement
from skcomplex.core.interactions import NeighborInteraction
from skcomplex.core.laws import DissipatorLaw, GeneratorLaw
from skcomplex.core.laws.utils import local_operator
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import OperatorCompiler, compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import JumpTrajectoriesSolver, LindbladSolver, UnitarySolver
from skcomplex.errors import SolverError
from skcomplex.models.physics.common.quantum import QUBIT, QubitElement, SpinElement
from skcomplex.specs import HilbertSpec, ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec

OMEGA = 2.0


GAMMA = 0.5


class RabiElement(QubitElement):
    """Qubit driven at the Rabi frequency `omega`."""

    def __init__(
        self,
        label: str,
        psi: Annotated[object, StateSpec(domain=CC)] = 'up',
        *,
        hilbert: HilbertSpec = QUBIT,
        visual: VisualSpec | None = None,
        omega: Annotated[float, ParameterSpec()] = OMEGA,
    ) -> None:
        super().__init__(label, psi, hilbert=hilbert, visual=visual)
        self.omega = omega

    def _get_laws(self, s):
        """Return the drive of the qubit."""
        return [GeneratorLaw(s.omega / 2 * local_operator(s.psi, 'x'))]


class DecayingElement(QubitElement):
    """Qubit that decays to its ground state at the rate `gamma`."""

    def __init__(
        self,
        label: str,
        psi: Annotated[object, StateSpec(domain=CC)] = 'up',
        *,
        hilbert: HilbertSpec = QUBIT,
        visual: VisualSpec | None = None,
        gamma: Annotated[float, ParameterSpec()] = GAMMA,
    ) -> None:
        super().__init__(label, psi, hilbert=hilbert, visual=visual)
        self.gamma = gamma

    def _get_laws(self, s):
        """Return the decay of the qubit."""
        return [DissipatorLaw(local_operator(s.psi, 'minus'), s.gamma)]


class DephasedElement(QubitElement):
    """Driven qubit that loses its coherence."""

    def _get_laws(self, s):
        """Return the drive and the dephasing of the qubit."""
        return [GeneratorLaw(local_operator(s.psi, 'x')), DissipatorLaw(local_operator(s.psi, 'z'), 0.3)]


class XXZInteraction(NeighborInteraction):
    """Anisotropic exchange."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        delta: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.delta = delta

    def _get_laws(self, s):
        """Return the exchange of one bond."""
        a, b = s.source.psi, s.target.psi
        return [
            GeneratorLaw(
                local_operator(a, 'x') * local_operator(b, 'x')
                + local_operator(a, 'y') * local_operator(b, 'y')
                + s.delta * local_operator(a, 'z') * local_operator(b, 'z'),
            ),
        ]


def qubit_z(s):
    """Return the `z` operator of the qubit."""
    return local_operator(s.q.psi, 'z')


def total_z(s):
    """Return the total magnetisation of the chain."""
    return sum(local_operator(member.psi, 'z') for member in s.chain)


Z = [ObservableSpec('z', qubit_z)]


MAGNETIZATION = [ObservableSpec('magnetization', total_z)]


def _chain(n_sites, **values):
    """Return a Heisenberg chain from a Néel state."""
    labels = ['up', 'down'] * (n_sites // 2)
    return System(
        elements=[
            LatticeElement('chain', element=SpinElement, space=LatticeSpace(n_sites), values={'psi': labels}),
        ],
        interactions=[XXZInteraction('xxz', target='chain', **values)],
    )


def test_unitary_evolution_follows_rabi_oscillations():
    """Test that a driven qubit oscillates as cos(omega t) with a constant norm."""
    system = System(elements=[RabiElement('q')])
    simulation = (
        Simulation(system, solver=UnitarySolver(), dt=0.05, observables=Z, compiler=OperatorCompiler())
        .prepare()
        .run(until=np.pi / OMEGA)
    )
    trajectory = simulation.trajectory_
    assert np.allclose(trajectory.values('z'), np.cos(OMEGA * trajectory.times), atol=1e-6)
    assert np.allclose(np.linalg.norm(trajectory.values('system.state'), axis=1), 1, atol=1e-8)


def test_unitary_evolution_conserves_magnetization_and_energy():
    """Test that an isotropic chain of 10 spins conserves its magnetisation and energy."""
    trajectory = (
        Simulation(
            _chain(10),
            observables=MAGNETIZATION,
            solver=UnitarySolver(atol=1e-14, rtol=1e-13),
            dt=1.0,
            compiler=OperatorCompiler(),
        )
        .prepare()
        .run(until=10.0)
        .trajectory_
    )
    assert np.ptp(trajectory.values('magnetization')) < 1e-8
    assert np.ptp(trajectory.values('energy')) < 1e-8


def test_unitary_evolution_rejects_open_systems():
    """Test that unitary evolution refuses dissipators."""
    with pytest.raises(SolverError, match='dissipators'):
        Simulation(
            System(elements=[DecayingElement('q')]),
            solver=UnitarySolver(),
            compiler=OperatorCompiler(),
        ).prepare().run(until=1.0)


def test_lindblad_evolution_follows_spontaneous_decay():
    """Test that the excited population decays as exp(-gamma t) with trace 1."""
    system = System(elements=[DecayingElement('q')])
    simulation = Simulation(system, dt=0.5, observables=Z, compiler=OperatorCompiler()).prepare().run(until=4.0)
    assert isinstance(simulation.stepper_, LindbladSolver)
    trajectory = simulation.trajectory_
    assert np.allclose((1 + trajectory.values('z')) / 2, np.exp(-GAMMA * trajectory.times), atol=1e-6)
    traces = np.trace(trajectory.values('system.state'), axis1=1, axis2=2)
    assert np.allclose(traces, 1, atol=1e-10)


def test_lindblad_evolution_lowers_purity():
    """Test that dephasing lowers the purity from 1 frame after frame."""
    trajectory = (
        Simulation(System(elements=[DephasedElement('q')]), dt=0.2, compiler=OperatorCompiler())
        .prepare()
        .run(until=4.0)
        .trajectory_
    )
    purity = trajectory.values('purity')
    assert purity[0] == pytest.approx(1)
    assert np.all(np.diff(purity) <= 1e-10)


def test_jump_trajectories_are_reproducible():
    """Test that two seeded trajectories of a decaying qubit are identical."""
    system = System(elements=[DecayingElement('q')])
    runs = [
        (
            Simulation(
                system,
                solver=JumpTrajectoriesSolver(),
                dt=0.5,
                random_state=3,
                observables=Z,
                compiler=OperatorCompiler(),
            )
            .prepare()
            .run(until=10.0)
        )
        for _ in range(2)
    ]
    assert np.array_equal(runs[0].trajectory_.values('z'), runs[1].trajectory_.values('z'))
    assert set(np.round(runs[0].trajectory_.values('z'), 8)) <= {-1.0, 1.0}


def test_jump_trajectories_average_to_the_master_equation():
    """Test that 500 trajectories average to the Lindblad decay within three standard errors."""
    system = System(elements=[DecayingElement('q')])
    problem = compile_system(system, observables=Z, compiler=OperatorCompiler())
    rng = np.random.default_rng(0)
    values = np.array(
        [
            Simulation(
                problem,
                solver=JumpTrajectoriesSolver(),
                dt=0.5,
                random_state=rng,
                compiler=OperatorCompiler(),
            )
            .prepare()
            .run(until=4.0)
            .trajectory_.values('z')
            for _ in range(500)
        ],
    )
    excited = (1 + values) / 2
    times = np.arange(9) * 0.5
    errors = excited.std(axis=0) / np.sqrt(len(excited)) + 1e-12
    assert np.all(np.abs(excited.mean(axis=0) - np.exp(-GAMMA * times)) <= 3 * errors + 1e-9)


def test_operator_solvers_follow_parameter_updates():
    """Test that a quench of the anisotropy changes the energy after the update."""
    simulation = (
        Simulation(_chain(6), observables=MAGNETIZATION, dt=0.5, compiler=OperatorCompiler()).prepare().run(until=1.0)
    )
    simulation.update_parameters({'xxz.delta': 0.0})
    trajectory = simulation.extend(until=2.0).trajectory_
    energy = trajectory.values('energy')
    assert np.ptp(energy[:3]) < 1e-8
    assert abs(energy[3] - energy[2]) > 0.1
    assert np.ptp(energy[3:]) < 1e-8
