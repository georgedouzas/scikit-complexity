"""Test the compilation of quantum systems."""

from operator import attrgetter

import numpy as np
import pytest
import qutip

from skcomplex.core.base import Form, HilbertSpaceError, IncompatibleLawsError, ValidationError, parameter, symbol
from skcomplex.core.compile import ObservableEvaluator, QuantumProblem, compile_system
from skcomplex.core.model import (
    Dissipator,
    Exchange,
    Generator,
    LatticeSpace,
    Mode,
    QuantumSystem,
    Qubit,
    Rate,
    Spin,
    System,
    expectation,
)
from skcomplex.models.biology.ecology import Population


class Heisenberg(Exchange):
    """Isotropic exchange with a coupling."""

    J: float = parameter(1.0)

    def laws(self, s):
        """Return the exchange of one bond."""
        a, b = s.first, s.second
        return [Generator(s.J * (a.x * b.x + a.y * b.y + a.z * b.z))]


Plus = Qubit.define('Plus', laws=lambda s: [Generator(s.plus)])
Foreign = Qubit.define('Foreign', laws=lambda s: [Generator(symbol('n_rabbits') * s.z)])
Classical = Qubit.define('Classical', laws=lambda s: [Rate(symbol('x'), 1)])
Decaying = Qubit.define('Decaying', parameters={'gamma': 0.5}, laws=lambda s: [Dissipator(s.minus, s.gamma)])
Negative = Qubit.define('Negative', laws=lambda s: [Dissipator(s.minus, -1.0)])


def local_z(member):
    """Return the local z operator of a member."""
    return member.z


def test_compile_system_gives_the_dimensions_of_a_chain_and_a_mode():
    """Test that a chain of spins and a mode compile to the tensor product of their spaces."""
    system = QuantumSystem(components=[Spin.lattice('chain', LatticeSpace(10)), Mode('cavity', quanta=4)])
    problem = compile_system(system)
    assert isinstance(problem, QuantumProblem)
    assert problem.form is Form.QUANTUM
    assert list(problem.dimensions) == [2] * 10 + [5]
    assert problem.dimension == 5120


def test_compile_system_rejects_too_large_spaces():
    """Test that a space above the dimension limit raises an error naming the group, its dimension, and the limit."""
    system = QuantumSystem(components=[Spin.group('chain', 40)], max_dimension=2**24)
    with pytest.raises(HilbertSpaceError, match=r'chain of dimension 1099511627776.*max_dimension = 16777216'):
        compile_system(system)


def test_compile_system_counts_one_exchange_term_per_bond():
    """Test that an exchange on a periodic lattice of 8 spins equals the sum of the 8 bond terms."""
    system = QuantumSystem(
        components=[Spin.lattice('chain', LatticeSpace(8))],
        interactions=[Heisenberg('h', target='chain')],
    )
    problem = compile_system(system)
    matrix = problem.hamiltonian_at(0.0, problem.initial_parameters).toarray()
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
    system = System(components=[Qubit.define('Fixed', laws=lambda s: [Generator(0.5 * s.z + s.x)])('q')])
    problem = compile_system(system)
    assert len(problem.hamiltonian) == 1
    assert problem.hamiltonian[0].function is None


def test_compile_system_rejects_non_hermitian_hamiltonians():
    """Test that a Hamiltonian without its adjoint raises an error that names the law."""
    with pytest.raises(ValidationError, match=r'Hamiltonian of q is not Hermitian.*adjoint'):
        compile_system(System(components=[Plus('q')]))


def test_compile_system_rejects_classical_symbols_in_coefficients():
    """Test that a coefficient reading a symbol other than parameters and time raises an error."""
    with pytest.raises(ValidationError, match='n_rabbits'):
        compile_system(System(components=[Foreign('q')]))


def test_compile_system_rejects_classical_elements_and_laws():
    """Test that quantum systems hold neither classical elements nor classical laws."""
    with pytest.raises(IncompatibleLawsError, match='rabbits'):
        compile_system(System(components=[Qubit('q'), Population('rabbits')]))
    with pytest.raises(IncompatibleLawsError, match='Rate'):
        compile_system(System(components=[Classical('q')]))


def test_compile_system_rejects_negative_rates():
    """Test that a dissipator with a negative rate raises an error."""
    with pytest.raises(ValidationError, match='non-negative'):
        compile_system(System(components=[Negative('q')]))


def test_compile_system_stores_neel_states():
    """Test that a Néel state has norm 1 and one nonzero amplitude at its basis index."""
    labels = ['up', 'down'] * 3
    problem = compile_system(QuantumSystem(components=[Spin.lattice('chain', LatticeSpace(6), initial=labels)]))
    state = problem.state_of(problem.initial_state)
    assert np.linalg.norm(state) == pytest.approx(1)
    assert np.flatnonzero(state).tolist() == [int('010101', 2)]


@pytest.mark.parametrize(('initial', 'message'), [(np.array([2.0, 0.0]), 'norm 2'), ('hot', 'ground')])
def test_compile_system_rejects_invalid_initial_states(initial, message):
    """Test that an unnormalised vector or an unknown name raises an error that names the system."""
    with pytest.raises(ValidationError, match=message):
        compile_system(QuantumSystem('box', components=[Qubit('q')], initial=initial))


def test_compile_system_stores_density_matrices_of_open_systems():
    """Test that an open system and a mixed initial state compile to density matrices."""
    problem = compile_system(System(components=[Decaying('q')]))
    assert not problem.pure
    assert problem.state_of(problem.initial_state).shape == (2, 2)
    mixed = compile_system(QuantumSystem(components=[Qubit('q')], initial=np.eye(2) / 2))
    assert not mixed.pure


def test_compile_system_starts_from_the_ground_state():
    """Test that the initial state 'ground' is the lowest eigenstate."""
    system = QuantumSystem(
        components=[Spin.lattice('chain', LatticeSpace(4))],
        interactions=[Heisenberg('h', target='chain')],
        initial='ground',
    )
    problem = compile_system(system)
    energy = ObservableEvaluator(problem).evaluate(0.0, problem.initial_state, problem.initial_parameters)['energy']
    assert energy == pytest.approx(-2.0)


def test_observable_evaluator_reads_local_magnetization():
    """Test that a group observable gives one expectation of the local z per site."""
    labels = ['up', 'down'] * 4
    system = QuantumSystem(
        components=[Spin.lattice('chain', LatticeSpace(8), initial=labels)],
    )
    observables = [
        expectation('sz', local_z, group='chain'),
        expectation('total', local_z, group='chain', reduce='sum'),
    ]
    problem = compile_system(system, observables=observables)
    values = ObservableEvaluator(problem).evaluate(0.0, problem.initial_state, problem.initial_parameters)
    assert values['sz'].tolist() == [0.5, -0.5] * 4
    assert values['total'] == 0.0


def test_observable_evaluator_rejects_non_operators():
    """Test that a quantum observable must give an operator."""
    system = QuantumSystem(components=[Qubit('q')])
    with pytest.raises(ValidationError, match='must give an operator'):
        ObservableEvaluator(compile_system(system, observables=[expectation('label', attrgetter('t'))]))


def test_local_values_give_probabilities_and_expectations():
    """Test the local values that visuals draw for single elements."""
    problem = compile_system(System(components=[Qubit('q', initial='down')]))
    values = problem.local_values('q', problem.initial_state)
    assert values['z'] == -1.0
    assert values['x'] == 0.0
