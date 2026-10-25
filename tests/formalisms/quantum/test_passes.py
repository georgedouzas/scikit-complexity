"""Test the compilation of quantum systems."""

import math
from operator import attrgetter
from typing import ClassVar

import numpy as np
import pytest
import qutip

from skcomplex.core.base import (
    BaseInteraction,
    Form,
    IncompatibleLawsError,
    LawKind,
    ValidationError,
    parameter,
    role,
    symbol,
)
from skcomplex.core.laws import RateLaw
from skcomplex.core.problems import BaseProblem, Problem
from skcomplex.core.space import LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import (
    BasePass,
    BaseSystemCompiler,
    ObservableEvaluator,
    claiming_compiler,
    compile_system,
)
from skcomplex.formalisms import registered_formalisms
from skcomplex.formalisms.quantum import (
    DissipatorLaw,
    GeneratorLaw,
    HilbertSpaceError,
    LindbladSolver,
    QuantumCompiler,
    QuantumProblem,
    expectation,
)
from skcomplex.formalisms.quantum._passes import hamiltonian_at, state_of
from skcomplex.models.biology.ecology import PopulationComponent
from skcomplex.models.physics.realizations.quantum import ModeComponent, QubitComponent, SpinComponent


class HeisenbergInteraction(BaseInteraction):
    """Isotropic exchange with a coupling."""

    over: ClassVar[str] = 'neighbors'
    target: str = role()

    J: float = parameter(1.0)

    def laws(self, s):
        """Return the exchange of one bond."""
        a, b = s.first, s.second
        return [GeneratorLaw(s.J * (a.x * b.x + a.y * b.y + a.z * b.z))]


PlusComponent = QubitComponent.define('PlusComponent', laws=lambda s: [GeneratorLaw(s.plus)])
ForeignComponent = QubitComponent.define('ForeignComponent', laws=lambda s: [GeneratorLaw(symbol('n_rabbits') * s.z)])
ClassicalComponent = QubitComponent.define('ClassicalComponent', laws=lambda s: [RateLaw(symbol('x'), 1)])
DecayingComponent = QubitComponent.define(
    'DecayingComponent',
    parameters={'gamma': 0.5},
    laws=lambda s: [DissipatorLaw(s.minus, s.gamma)],
)
NegativeComponent = QubitComponent.define('NegativeComponent', laws=lambda s: [DissipatorLaw(s.minus, -1.0)])


def local_z(member):
    """Return the local z operator of a member."""
    return member.z


def test_compile_system_gives_the_dimensions_of_a_chain_and_a_mode():
    """Test that a chain of spins and a mode compile to the tensor product of their spaces."""
    system = System(components=[SpinComponent.lattice('chain', LatticeSpace(10)), ModeComponent('cavity', quanta=4)])
    problem = compile_system(system)
    assert isinstance(problem, QuantumProblem)
    assert problem.form == Form.QUANTUM
    assert list(problem.dimensions) == [2] * 10 + [5]
    assert math.prod(problem.dimensions) == 5120


def test_compile_system_rejects_too_large_spaces():
    """Test that a space above the dimension limit raises an error naming the group, its dimension, and the limit."""
    system = System(components=[SpinComponent.group('chain', 40)])
    with pytest.raises(HilbertSpaceError, match=r'chain of dimension 1099511627776.*max_dimension = 16777216'):
        compile_system(system, solver=LindbladSolver(max_dimension=2**24))


def test_compile_system_counts_one_exchange_term_per_bond():
    """Test that an exchange on a periodic lattice of 8 spins equals the sum of the 8 bond terms."""
    system = System(
        components=[SpinComponent.lattice('chain', LatticeSpace(8))],
        interactions=[HeisenbergInteraction('h', target='chain')],
    )
    problem = compile_system(system)
    matrix = hamiltonian_at(problem, 0.0, problem.initial_parameters).toarray()
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
        components=[QubitComponent.define('FixedComponent', laws=lambda s: [GeneratorLaw(0.5 * s.z + s.x)])('q')],
    )
    problem = compile_system(system)
    assert len(problem.hamiltonian) == 1
    assert problem.hamiltonian[0].function is None


def test_compile_system_rejects_non_hermitian_hamiltonians():
    """Test that a Hamiltonian without its adjoint raises an error that names the law."""
    with pytest.raises(ValidationError, match=r'Hamiltonian of q is not Hermitian.*adjoint'):
        compile_system(System(components=[PlusComponent('q')]))


def test_compile_system_rejects_classical_symbols_in_coefficients():
    """Test that a coefficient reading a symbol other than parameters and time raises an error."""
    with pytest.raises(ValidationError, match='n_rabbits'):
        compile_system(System(components=[ForeignComponent('q')]))


def test_compile_system_rejects_classical_elements_and_laws():
    """Test that quantum systems hold neither classical elements nor classical laws."""
    with pytest.raises(IncompatibleLawsError, match='rabbits'):
        compile_system(System(components=[QubitComponent('q'), PopulationComponent('rabbits')]))
    with pytest.raises(IncompatibleLawsError, match='RateLaw'):
        compile_system(System(components=[ClassicalComponent('q')]))


def test_compile_system_rejects_negative_rates():
    """Test that a dissipator with a negative rate raises an error."""
    with pytest.raises(ValidationError, match='non-negative'):
        compile_system(System(components=[NegativeComponent('q')]))


def test_compile_system_stores_neel_states():
    """Test that a Néel state has norm 1 and one nonzero amplitude at its basis index."""
    labels = ['up', 'down'] * 3
    problem = compile_system(System(components=[SpinComponent.lattice('chain', LatticeSpace(6), initial=labels)]))
    state = state_of(problem, problem.initial_state)
    assert np.linalg.norm(state) == pytest.approx(1)
    assert np.flatnonzero(state).tolist() == [int('010101', 2)]


@pytest.mark.parametrize(('initial', 'message'), [(np.array([2.0, 0.0]), 'norm 2'), ('hot', 'ground')])
def test_compile_system_rejects_invalid_initial_states(initial, message):
    """Test that an unnormalised vector or an unknown name raises an error that names the system."""
    with pytest.raises(ValidationError, match=message):
        compile_system(System('box', components=[QubitComponent('q')]), initial_state={'box.state': initial})


def test_compile_system_stores_density_matrices_of_open_systems():
    """Test that an open system and a mixed initial state compile to density matrices."""
    problem = compile_system(System(components=[DecayingComponent('q')]))
    assert not problem.pure
    assert state_of(problem, problem.initial_state).shape == (2, 2)
    mixed = compile_system(System(components=[QubitComponent('q')]), initial_state={'system.state': np.eye(2) / 2})
    assert not mixed.pure


def test_compile_system_starts_from_the_ground_state():
    """Test that the initial state 'ground' is the lowest eigenstate."""
    system = System(
        components=[SpinComponent.lattice('chain', LatticeSpace(4))],
        interactions=[HeisenbergInteraction('h', target='chain')],
    )
    problem = compile_system(system, initial_state={'system.state': 'ground'})
    energy = (
        ObservableEvaluator(problem)
        .prepare()
        .evaluate(0.0, problem.initial_state, problem.initial_parameters)['energy']
    )
    assert energy == pytest.approx(-2.0)


def test_observable_evaluator_reads_local_magnetization():
    """Test that a group observable gives one expectation of the local z per site."""
    labels = ['up', 'down'] * 4
    system = System(components=[SpinComponent.lattice('chain', LatticeSpace(8), initial=labels)])
    observables = [
        expectation('sz', local_z, group='chain'),
        expectation('total', local_z, group='chain', reduce='sum'),
    ]
    problem = compile_system(system, observables=observables)
    values = ObservableEvaluator(problem).prepare().evaluate(0.0, problem.initial_state, problem.initial_parameters)
    assert values['sz'].tolist() == [0.5, -0.5] * 4
    assert values['total'] == 0.0


def test_observable_evaluator_rejects_non_operators():
    """Test that a quantum observable must give an operator."""
    system = System(components=[QubitComponent('q')])
    with pytest.raises(ValidationError, match='must give an operator'):
        ObservableEvaluator(compile_system(system, observables=[expectation('label', attrgetter('t'))])).prepare()


def test_quantum_pass_claims_quantum_systems():
    """Test that the registered quantum pass claims a quantum system and leaves a classical one."""
    assert isinstance(claiming_compiler(System(components=[SpinComponent.group('chain', 2)])), QuantumCompiler)
    assert claiming_compiler(System(components=[PopulationComponent('rabbits', n=10)])) is None


def test_registered_formalisms_list_quantum():
    """Test that the quantum formalism is listed with its law kinds, its form, and its solvers."""
    quantum = next(item for item in registered_formalisms() if item.name == 'quantum')
    assert {law.law_kind for law in quantum.laws} == {LawKind.GENERATOR, LawKind.DISSIPATOR}
    assert quantum.forms == (Form.QUANTUM,)
    assert quantum.defaults == (LindbladSolver,)


def test_quantum_compiler_is_a_system_compiler_and_not_a_pass():
    """Test that the quantum compiler compiles whole systems and is not a step of the pipeline of passes."""
    assert issubclass(QuantumCompiler, BaseSystemCompiler)
    assert not issubclass(QuantumCompiler, BasePass)


def test_quantum_problem_is_a_base_problem_and_not_the_compilers():
    """Test that a quantum problem derives from the problem base alone, and that its run is recorded."""
    assert issubclass(QuantumProblem, BaseProblem)
    assert not issubclass(QuantumProblem, Problem)
