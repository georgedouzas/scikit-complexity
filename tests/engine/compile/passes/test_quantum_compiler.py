"""Test the `QuantumCompiler` class."""

from operator import attrgetter
from typing import Annotated

import numpy as np
import pytest
import qutip
from sage.all__sagemath_symbolics import CC

from skcomplex.core.elements import GroupElement, LatticeElement
from skcomplex.core.interactions import NeighborInteraction
from skcomplex.core.laws import DissipatorLaw, GeneratorLaw, RateLaw
from skcomplex.core.laws.utils import local_operator
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine._utils import symbol
from skcomplex.engine.compile import (
    BaseCompiler,
    BasePass,
    ObservableEvaluator,
    QuantumCompiler,
    compile_system,
)
from skcomplex.errors import IncompatibleLawsError, ValidationError
from skcomplex.models.biology.ecology import PopulationElement
from skcomplex.models.physics.common.quantum import (
    QUBIT,
    QubitElement,
    SpinElement,
    spin_space,
)
from skcomplex.specs import HilbertSpec, ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec


class HeisenbergInteraction(NeighborInteraction):
    """Isotropic exchange with a coupling."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        J: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.J = J

    def _get_laws(self, s):
        """Return the exchange of one bond."""
        a, b = s.source.psi, s.target.psi
        exchange = sum(local_operator(a, axis) * local_operator(b, axis) for axis in 'xyz')
        return [GeneratorLaw(s.J * exchange)]


class FixedElement(QubitElement):
    """Qubit with a constant Hamiltonian."""

    def _get_laws(self, s):
        """Return the constant Hamiltonian."""
        return [GeneratorLaw(0.5 * local_operator(s.psi, 'z') + local_operator(s.psi, 'x'))]


class PlusElement(QubitElement):
    """Qubit whose Hamiltonian is a ladder operator without its adjoint."""

    def _get_laws(self, s):
        """Return the non-Hermitian Hamiltonian."""
        return [GeneratorLaw(local_operator(s.psi, 'plus'))]


class ForeignElement(QubitElement):
    """Qubit whose Hamiltonian reads a classical symbol."""

    def _get_laws(self, s):
        """Return the Hamiltonian with a classical coefficient."""
        return [GeneratorLaw(symbol('n_rabbits') * local_operator(s.psi, 'z'))]


class ClassicalElement(QubitElement):
    """Qubit with a classical rate law."""

    def _get_laws(self, s):
        """Return the classical rate law."""
        return [RateLaw(symbol('x'), 1)]


class DecayingElement(QubitElement):
    """Qubit that decays at a rate."""

    def __init__(
        self,
        label: str,
        psi: Annotated[object, StateSpec(domain=CC)] = 'up',
        *,
        hilbert: HilbertSpec = QUBIT,
        visual: VisualSpec | None = VisualSpec('expectations'),
        gamma: Annotated[float, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, psi, hilbert=hilbert, visual=visual)
        self.gamma = gamma

    def _get_laws(self, s):
        """Return the decay."""
        return [DissipatorLaw(local_operator(s.psi, 'minus'), s.gamma)]


class NegativeElement(QubitElement):
    """Qubit with a negative decay rate."""

    def _get_laws(self, s):
        """Return the decay at a negative rate."""
        return [DissipatorLaw(local_operator(s.psi, 'minus'), -1.0)]


def local_z(member):
    """Return the local z operator of a member."""
    return local_operator(member.psi, 'z')


def test_compile_system_counts_one_exchange_term_per_bond():
    """Test that an exchange on a periodic lattice of 8 spins equals the sum of the 8 bond terms."""
    system = System(
        elements=[
            LatticeElement('chain', element=SpinElement, space=LatticeSpace(8), values={'hilbert': spin_space()}),
        ],
        interactions=[HeisenbergInteraction('h', target='chain')],
    )
    problem = compile_system(system, compiler=QuantumCompiler())
    matrix = problem._hamiltonian_at(0.0, problem.initial_parameters).toarray()
    spins = [[qutip.jmat(0.5, axis) for axis in 'xyz'] for _ in range(8)]
    expected = 0
    for site in range(8):
        for axis in range(3):
            factors = [qutip.qeye(2)] * 8
            factors[site], factors[(site + 1) % 8] = spins[site][axis], spins[site][axis]
            expected = expected + qutip.tensor(*factors)
    assert np.allclose(matrix, expected.full())
    assert len(problem.hamiltonian) == 1


def test_compile_system_folds_constant_hamiltonians():
    """Test that a Hamiltonian with numeric coefficients folds to one constant matrix."""
    system = System(
        elements=[
            FixedElement('q'),
        ],
    )
    problem = compile_system(system, compiler=QuantumCompiler())
    assert len(problem.hamiltonian) == 1
    assert problem.hamiltonian[0].function is None


def test_compile_system_rejects_non_hermitian_hamiltonians():
    """Test that a Hamiltonian without its adjoint raises an error that names the law."""
    with pytest.raises(ValidationError, match=r'Hamiltonian of q is not Hermitian.*adjoint'):
        compile_system(System(elements=[PlusElement('q')]), compiler=QuantumCompiler())


def test_compile_system_rejects_classical_symbols_in_coefficients():
    """Test that a coefficient reading a symbol other than parameters and time raises an error."""
    with pytest.raises(ValidationError, match='n_rabbits'):
        compile_system(System(elements=[ForeignElement('q')]), compiler=QuantumCompiler())


def test_compile_system_rejects_classical_elements_and_laws():
    """Test that quantum systems hold neither classical elements nor classical laws."""
    with pytest.raises(IncompatibleLawsError, match='rabbits'):
        compile_system(System(elements=[QubitElement('q'), PopulationElement('rabbits')]), compiler=QuantumCompiler())
    with pytest.raises(IncompatibleLawsError, match='RateLaw'):
        compile_system(System(elements=[ClassicalElement('q')]), compiler=QuantumCompiler())


def test_compile_system_rejects_negative_rates():
    """Test that a dissipator with a negative rate raises an error."""
    with pytest.raises(ValidationError, match='non-negative'):
        compile_system(System(elements=[NegativeElement('q')]), compiler=QuantumCompiler())


def test_compile_system_stores_neel_states():
    """Test that a Néel state has norm 1 and one nonzero amplitude at its basis index."""
    labels = ['up', 'down'] * 3
    problem = compile_system(
        System(
            elements=[
                LatticeElement('chain', element=SpinElement, space=LatticeSpace(6), values={'psi': labels}),
            ],
        ),
        compiler=QuantumCompiler(),
    )
    state = problem._state_of(problem.initial_state)
    assert np.linalg.norm(state) == pytest.approx(1)
    assert np.flatnonzero(state).tolist() == [int('010101', 2)]


@pytest.mark.parametrize(('initial', 'message'), [(np.array([2.0, 0.0]), 'norm 2'), ('hot', 'ground')])
def test_compile_system_rejects_invalid_initial_states(initial, message):
    """Test that an unnormalised vector or an unknown name raises an error that names the system."""
    with pytest.raises(ValidationError, match=message):
        compile_system(
            System('box', elements=[QubitElement('q')]),
            initial_state={'box.state': initial},
            compiler=QuantumCompiler(),
        )


def test_compile_system_stores_density_matrices_of_open_systems():
    """Test that an open system and a mixed initial state compile to density matrices."""
    problem = compile_system(System(elements=[DecayingElement('q')]), compiler=QuantumCompiler())
    assert not problem.pure
    assert problem._state_of(problem.initial_state).shape == (2, 2)
    mixed = compile_system(
        System(elements=[QubitElement('q')]), initial_state={'system.state': np.eye(2) / 2}, compiler=QuantumCompiler(),
    )
    assert not mixed.pure


def test_compile_system_starts_from_the_ground_state():
    """Test that the initial state 'ground' is the lowest eigenstate."""
    system = System(
        elements=[
            LatticeElement('chain', element=SpinElement, space=LatticeSpace(4), values={'hilbert': spin_space()}),
        ],
        interactions=[HeisenbergInteraction('h', target='chain')],
    )
    problem = compile_system(system, initial_state={'system.state': 'ground'}, compiler=QuantumCompiler())
    energy = (
        ObservableEvaluator(problem)
        .prepare()
        .evaluate(0.0, problem.initial_state, problem.initial_parameters)['energy']
    )
    assert energy == pytest.approx(-2.0)


def test_observable_evaluator_reads_local_magnetization():
    """Test that a group observable gives one expectation of the local z per site, and their sum."""
    labels = ['up', 'down'] * 4
    system = System(
        elements=[
            LatticeElement(
                'chain',
                element=SpinElement,
                space=LatticeSpace(8),
                values={'psi': labels, 'hilbert': spin_space()},
            ),
        ],
    )
    observables = [
        ObservableSpec('sz', local_z, group='chain', reduce=np.asarray),
        ObservableSpec('total', local_z, group='chain', reduce='sum'),
    ]
    problem = compile_system(system, observables=observables, compiler=QuantumCompiler())
    values = ObservableEvaluator(problem).prepare().evaluate(0.0, problem.initial_state, problem.initial_parameters)
    assert values['sz'].tolist() == [0.5, -0.5] * 4
    assert values['total'] == 0.0


def test_observable_evaluator_rejects_non_operators():
    """Test that a quantum observable must give an operator."""
    system = System(elements=[QubitElement('q')])
    with pytest.raises(ValidationError, match='must give an operator'):
        ObservableEvaluator(
            compile_system(system, observables=[ObservableSpec('label', attrgetter('t'))], compiler=QuantumCompiler()),
        ).prepare()


def test_quantum_compiler_reports_a_classical_system():
    """Test that the quantum compiler accepts a quantum system and reports a classical one."""
    compiler = QuantumCompiler().check()
    assert compiler.prepare(System(elements=[GroupElement('chain', element=SpinElement, size=2)])).report_.ok
    assert not compiler.prepare(System(elements=[PopulationElement('rabbits', n=10)])).report_.ok


def test_quantum_compiler_is_a_system_compiler_and_not_a_pass():
    """Test that the quantum compiler compiles whole systems and is not a step of the pipeline of passes."""
    assert issubclass(QuantumCompiler, BaseCompiler)
    assert not issubclass(QuantumCompiler, BasePass)
