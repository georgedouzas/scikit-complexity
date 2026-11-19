"""Test the `UnitarySolver` class."""

from types import SimpleNamespace

import numpy as np
import pytest

from skcomplex.core.components import Observable
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws.utils import local_operator
from skcomplex.core.components.spaces import BaseSpace, LatticeSpace
from skcomplex.core.engine.solvers import BaseSolver
from skcomplex.core.errors import SolverError
from skcomplex.core.specs import Role
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.quantum import (
    DissipatorLaw,
    GeneratorLaw,
    JumpTrajectoriesSolver,
    QubitElement,
    UnitarySolver,
)
from skcomplex.systems.physics.statistical_mechanics._quantum_spins import SpinElement

OMEGA = 2.0
GAMMA = 0.5


class RabiInteraction(Interaction):
    """Drive of a qubit at the Rabi frequency `omega`."""

    def __init__(self, label: str, *, target: Role, omega: float = OMEGA) -> None:
        super().__init__(label)
        self.target = target
        self.omega = omega

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the drive of the qubit."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [GeneratorLaw(context.omega / 2 * local_operator(context.target.psi, 'x'))]


class DecayInteraction(Interaction):
    """Decay of a qubit to its ground state at the rate `gamma`."""

    def __init__(self, label: str, *, target: Role, gamma: float = GAMMA) -> None:
        super().__init__(label)
        self.target = target
        self.gamma = gamma

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the decay of the qubit."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [DissipatorLaw(local_operator(context.target.psi, 'minus'), context.gamma)]


class DephasingInteraction(Interaction):
    """Drive of a qubit that loses its coherence."""

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the drive and the dephasing of the qubit."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            psi = context.target.psi
            self.laws_ = [GeneratorLaw(local_operator(psi, 'x')), DissipatorLaw(local_operator(psi, 'z'), 0.3)]


class XXZInteraction(Interaction):
    """Anisotropic exchange along the bonds of a chain."""

    def __init__(self, label: str, *, among: Role, along: bool = True, delta: float = 1.0) -> None:
        super().__init__(label, among=among, along=along)
        self.delta = delta

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the exchange of one bond."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            a, b = context.source.psi, context.target.psi
            self.laws_ = [
                GeneratorLaw(
                    local_operator(a, 'x') * local_operator(b, 'x')
                    + local_operator(a, 'y') * local_operator(b, 'y')
                    + context.delta * local_operator(a, 'z') * local_operator(b, 'z'),
                ),
            ]


def _qubit(interaction: type[Interaction], solver: BaseSolver | str = 'auto') -> System:
    """Return a qubit under one interaction, prepared with a solver."""
    return System(elements=[QubitElement('q')], interactions=[interaction('law', target='q')], solver=solver).prepare()


def qubit_z(s):
    """Return the `z` operator of the qubit."""
    return local_operator(s.q.psi, 'z')


Z = [Observable('z', qubit_z)]


def total_z(s):
    """Return the total magnetisation of the chain."""
    return sum(local_operator(member.psi, 'z') for member in s.chain)


MAGNETIZATION = [Observable('magnetization', total_z)]


def _chain(n_sites, **values):
    """Return a Heisenberg chain from a Néel state."""
    labels = ['up', 'down'] * (n_sites // 2)
    return System(
        elements=[SpinElement('chain', psi=labels, space=LatticeSpace(n_sites))],
        interactions=[XXZInteraction('xxz', among='chain', **values)],
    )


def test_unitary_evolution_follows_rabi_oscillations():
    """Test that a driven qubit oscillates as cos(omega t) with a constant norm."""
    trajectory = _qubit(RabiInteraction, UnitarySolver()).simulate(until=np.pi / OMEGA, dt=0.05, observables=Z)
    assert np.allclose(trajectory.get_values('z'), np.cos(OMEGA * trajectory.times), atol=1e-6)
    assert np.allclose(np.linalg.norm(trajectory.get_values('system.state'), axis=1), 1, atol=1e-8)


def test_unitary_evolution_conserves_magnetization_and_energy():
    """Test that an isotropic chain of 10 spins conserves its magnetisation and energy."""
    system = _chain(10).set_params(solver=UnitarySolver(atol=1e-14, rtol=1e-13)).prepare()
    trajectory = system.simulate(until=10.0, observables=MAGNETIZATION, dt=1.0)
    assert np.ptp(trajectory.get_values('magnetization')) < 1e-8
    assert np.ptp(trajectory.get_values('energy')) < 1e-8


def test_unitary_evolution_rejects_open_systems():
    """Test that unitary evolution refuses dissipators."""
    with pytest.raises(SolverError, match='dissipators'):
        _qubit(DecayInteraction, UnitarySolver()).simulate(until=1.0)


def test_lindblad_evolution_follows_spontaneous_decay():
    """Test that the excited population decays as exp(-gamma t) with trace 1."""
    trajectory = _qubit(DecayInteraction).simulate(until=4.0, dt=0.5, observables=Z)
    assert np.allclose((1 + trajectory.get_values('z')) / 2, np.exp(-GAMMA * trajectory.times), atol=1e-6)
    traces = np.trace(trajectory.get_values('system.state'), axis1=1, axis2=2)
    assert np.allclose(traces, 1, atol=1e-10)


def test_lindblad_evolution_lowers_purity():
    """Test that dephasing lowers the purity from 1 frame after frame."""
    purity = _qubit(DephasingInteraction).simulate(until=4.0, dt=0.2).get_values('purity')
    assert purity[0] == pytest.approx(1)
    assert np.all(np.diff(purity) <= 1e-10)


def test_jump_trajectories_are_reproducible():
    """Test that two seeded trajectories of a decaying qubit are identical."""
    system = _qubit(DecayInteraction, JumpTrajectoriesSolver())
    runs = [system.simulate(until=10.0, dt=0.5, random_state=3, observables=Z) for _ in range(2)]
    assert np.array_equal(runs[0].get_values('z'), runs[1].get_values('z'))
    assert set(np.round(runs[0].get_values('z'), 8)) <= {-1.0, 1.0}


def test_jump_trajectories_average_to_the_master_equation():
    """Test that 500 trajectories average to the Lindblad decay within three standard errors."""
    system = _qubit(DecayInteraction, JumpTrajectoriesSolver())
    rng = np.random.default_rng(0)
    values = np.array(
        [system.simulate(until=4.0, dt=0.5, random_state=rng, observables=Z).get_values('z') for _ in range(500)],
    )
    excited = (1 + values) / 2
    times = np.arange(9) * 0.5
    errors = excited.std(axis=0) / np.sqrt(len(excited)) + 1e-12
    assert np.all(np.abs(excited.mean(axis=0) - np.exp(-GAMMA * times)) <= 3 * errors + 1e-9)


def test_operator_solvers_follow_parameter_updates():
    """Test that a quench of the anisotropy changes the energy after the update."""
    trajectory = (
        _chain(6)
        .prepare()
        .simulate(
            until=2.0,
            observables=MAGNETIZATION,
            dt=0.5,
            schedule={'xxz.delta': lambda t: 1.0 if t < 1.0 else 0.0},
        )
    )
    energy = trajectory.get_values('energy')
    assert np.ptp(energy[:2]) < 1e-8
    assert abs(energy[2] - energy[1]) > 0.1
    assert np.ptp(energy[2:]) < 1e-8
