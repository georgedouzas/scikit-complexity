"""Test the `OperatorCompiler` class."""

import math
from operator import attrgetter
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
import qutip
from sage.all__sagemath_symbolics import CC

from skcomplex.core.components import Observable
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import local_operator
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.engine._utils import symbol
from skcomplex.core.engine.compile import BaseCompiler, BasePass
from skcomplex.core.engine.compile._observables import ObservableEvaluator
from skcomplex.core.engine.compile.passes._operator_compiler import OperatorCompiler
from skcomplex.core.engine.problems._operator_problem_spec import OperatorProblemSpec
from skcomplex.core.enums import FormKind
from skcomplex.core.errors import IncompatibleLawsError, ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role
from skcomplex.systems.biology.common.populations import PopulationElement
from skcomplex.systems.physics.common.quantum import (
    QUBIT,
    DissipatorLaw,
    GeneratorLaw,
    ModeElement,
    QuantumCompiler,
    QubitElement,
    mode_space,
)
from skcomplex.systems.physics.statistical_mechanics._quantum_spins import SPIN_HALF, SpinElement


class ExchangeInteraction(Interaction):
    """Isotropic exchange of every pair of neighbouring spins, set from the namespace of the pair."""

    def __init__(self, label: str, *, among: Role, J: float = 1.0, along: bool = True) -> None:
        super().__init__(label, among=among, along=along)
        self.J = J

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the exchange of one pair."""
        super()._prepare(context)
        a, b = context.source.psi, context.target.psi
        exchange = sum(local_operator(a, axis) * local_operator(b, axis) for axis in 'xyz')
        self.laws_ = [GeneratorLaw(context.J * exchange)]


class DecayInteraction(Interaction):
    """Decay of a qubit at a rate."""

    def __init__(self, label: str, *, target: Role = 'q', gamma: float = 0.5) -> None:
        super().__init__(label)
        self.target = target
        self.gamma = gamma

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the decay."""
        super()._prepare(context)
        self.laws_ = [DissipatorLaw(local_operator(context.target.psi, 'minus'), context.gamma)]


class RuleInteraction(Interaction):
    """Laws of a qubit, a function of its namespace."""

    def __init__(self, label: str, *, target: Role = 'q', rule=None) -> None:
        super().__init__(label)
        self.target = target
        self.rule = rule

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the laws the rule gives."""
        super()._prepare(context)
        self.laws_ = self.rule(context.target)


def _qubit(laws):
    """Return one qubit `q` whose laws are a function of its namespace."""
    return System(elements=[QubitElement('q')], interactions=[RuleInteraction('rule', rule=laws)])


def _fixed(q):
    """Return the constant generator of a qubit."""
    return [GeneratorLaw(0.5 * local_operator(q.psi, 'z') + local_operator(q.psi, 'x'))]


def _chain(size, psi='up', interactions=()):
    """Return a chain of spins one half on a periodic lattice."""
    chain = SpinElement('chain', psi=psi, hilbert=SPIN_HALF, space=LatticeSpace(size))
    return System(elements=[chain], interactions=list(interactions))


def local_z(member):
    """Return the local z operator of a member."""
    return local_operator(member.psi, 'z')


def test_compiler_counts_one_exchange_term_per_bond():
    """Test that an exchange on a periodic lattice of 8 spins equals the sum of the 8 bond terms."""
    system = _chain(8, interactions=[ExchangeInteraction('h', among='chain')])
    problem = QuantumCompiler().prepare(system).run().problem_
    matrix = problem._get_generator_at(0.0, problem.initial_parameters).toarray()
    spins = [[qutip.jmat(0.5, axis) for axis in 'xyz'] for _ in range(8)]
    expected = 0
    for site in range(8):
        for axis in range(3):
            factors = [qutip.qeye(2)] * 8
            factors[site], factors[(site + 1) % 8] = spins[site][axis], spins[site][axis]
            expected = expected + qutip.tensor(*factors)
    assert np.allclose(matrix, expected.full())
    assert len(problem.generator) == 1


def test_compiler_folds_constant_generators():
    """Test that a generator with numeric coefficients folds to one constant matrix."""
    problem = QuantumCompiler().prepare(_qubit(_fixed)).run().problem_
    assert len(problem.generator) == 1
    assert problem.generator[0].function is None


def test_compiler_rejects_non_hermitian_generators():
    """Test that a generator without its adjoint raises an error that names the law."""
    with pytest.raises(ValidationError, match=r'generator of rule is not Hermitian.*adjoint'):
        QuantumCompiler().prepare(_qubit(lambda q: [GeneratorLaw(local_operator(q.psi, 'plus'))])).run()


def test_compiler_rejects_classical_symbols_in_coefficients():
    """Test that a coefficient reading a symbol other than parameters and time raises an error."""
    foreign = _qubit(lambda q: [GeneratorLaw(symbol('n_rabbits') * local_operator(q.psi, 'z'))])
    with pytest.raises(ValidationError, match='n_rabbits'):
        QuantumCompiler().prepare(foreign).run()


def test_compiler_rejects_classical_elements_and_laws():
    """Test that operator systems hold neither classical elements nor classical laws."""
    with pytest.raises(IncompatibleLawsError, match='rabbits'):
        QuantumCompiler().prepare(System(elements=[QubitElement('q'), PopulationElement('rabbits')])).run()
    with pytest.raises(IncompatibleLawsError, match='RateLaw'):
        QuantumCompiler().prepare(_qubit(lambda q: [RateLaw(symbol('x'), 1)])).run()


def test_compiler_rejects_negative_rates():
    """Test that a dissipator with a negative rate raises an error."""
    with pytest.raises(ValidationError, match='non-negative'):
        QuantumCompiler().prepare(_qubit(lambda q: [DissipatorLaw(local_operator(q.psi, 'minus'), -1.0)])).run()


def test_compiler_stores_neel_states():
    """Test that a Néel state has norm 1 and one nonzero amplitude at its basis index."""
    problem = QuantumCompiler().prepare(_chain(6, ['up', 'down'] * 3)).run().problem_
    state = problem._get_state_of(problem.initial_state)
    assert np.linalg.norm(state) == pytest.approx(1)
    assert np.flatnonzero(state).tolist() == [int('010101', 2)]


@pytest.mark.parametrize(('initial', 'message'), [(np.array([2.0, 0.0]), 'norm 2'), ('hot', 'ground')])
def test_compiler_rejects_invalid_initial_states(initial, message):
    """Test that an unnormalised vector or an unknown name raises an error that names the system."""
    with pytest.raises(ValidationError, match=message):
        QuantumCompiler().prepare(System('box', elements=[QubitElement('q')])).run({'box.state': initial})


def test_compiler_stores_density_matrices_of_open_systems():
    """Test that an open system and a mixed initial state compile to density matrices."""
    system = System(elements=[QubitElement('q')], interactions=[DecayInteraction('decay')])
    problem = QuantumCompiler().prepare(system).run().problem_
    assert not problem.pure
    assert problem._get_state_of(problem.initial_state).shape == (2, 2)
    system = System(elements=[QubitElement('q')])
    mixed = QuantumCompiler().prepare(system).run({'system.state': np.eye(2) / 2}).problem_
    assert not mixed.pure


def test_compiler_starts_from_the_ground_state():
    """Test that the initial state 'ground' is the lowest eigenstate."""
    system = _chain(4, interactions=[ExchangeInteraction('h', among='chain')])
    problem = QuantumCompiler().prepare(system).run({'system.state': 'ground'}).problem_
    energy = (
        ObservableEvaluator(problem)
        .prepare()
        .evaluate(0.0, problem.initial_state, problem.initial_parameters)['energy']
    )
    assert energy == pytest.approx(-2.0)


def test_observable_evaluator_reads_local_magnetization():
    """Test that a group observable gives one expectation of the local z per site, and their sum."""
    system = _chain(8, ['up', 'down'] * 4)
    observables = [
        Observable('sz', local_z, group='chain', reduce=np.asarray),
        Observable('total', local_z, group='chain', reduce='sum'),
    ]
    problem = QuantumCompiler(observables=observables).prepare(system).run().problem_
    values = ObservableEvaluator(problem).prepare().evaluate(0.0, problem.initial_state, problem.initial_parameters)
    assert values['sz'].tolist() == [0.5, -0.5] * 4
    assert values['total'] == 0.0


def test_observable_evaluator_rejects_non_operators():
    """Test that an operator observable must give an operator."""
    system = System(elements=[QubitElement('q')])
    with pytest.raises(ValidationError, match='must give an operator'):
        ObservableEvaluator(
            QuantumCompiler(observables=[Observable('label', attrgetter('t'))]).prepare(system).run().problem_,
        ).prepare()


def test_operator_compiler_reports_a_classical_system():
    """Test that the operator compiler accepts an operator system and reports a classical one."""
    compiler = OperatorCompiler()
    assert compiler.prepare(_chain(2)).report_.ok
    assert not compiler.prepare(System(elements=[PopulationElement('rabbits', x=10)])).report_.ok


def test_operator_compiler_is_a_system_compiler_and_not_a_pass():
    """Test that the operator compiler compiles whole systems and is not a step of the pipeline of passes."""
    assert issubclass(OperatorCompiler, BaseCompiler)
    assert not issubclass(OperatorCompiler, BasePass)


def test_operator_compiler_get_hilbert_state():
    """Test `OperatorCompiler._get_hilbert_state` on a small case."""
    assert OperatorCompiler._get_hilbert_state(QubitElement) == 'psi'


def test_dimension_rejects_an_element_without_a_hilbert_space():
    """Test that a subsystem without a Hilbert space raises an error."""
    with pytest.raises(ValidationError, match='takes no Hilbert space'):
        _ = OperatorCompiler._get_view_of(SpinElement('p', hilbert=None)).dimension


def test_compiler_gives_the_dimensions_of_a_chain_and_a_mode():
    """Test that a chain of spins and a mode compile to the tensor product of their spaces."""
    system = System(
        elements=[
            SpinElement('chain', space=LatticeSpace(10)),
            ModeElement('cavity', hilbert=mode_space(4)),
        ],
    )
    problem = QuantumCompiler().prepare(system).run().problem_
    assert isinstance(problem, OperatorProblemSpec)
    assert problem.form == FormKind.OPERATOR
    assert list(problem.dimensions) == [2] * 10 + [5]
    assert math.prod(problem.dimensions) == 5120


def test_operator_compiler_leaves_every_law_to_the_compiler_of_a_field():
    """Test that the operator compiler compiles elements without laws and rejects any law it does not add."""
    problem = OperatorCompiler().prepare(System(elements=[QubitElement('q')])).run().problem_
    assert problem.dimensions == (2,)
    with pytest.raises(IncompatibleLawsError, match='GeneratorLaw, which OperatorCompiler cannot compile'):
        OperatorCompiler().prepare(_qubit(_fixed)).run()


class TwoStateElement(Element):
    """Plain two-state element whose state has complex allowed values."""

    def __init__(self, label: str, phi: Annotated[object, CC] = 'up', *, tag: str = 'a') -> None:
        super().__init__(label)
        self.phi = phi
        self.tag = tag
        self.hilbert = QUBIT


def test_compiler_couples_the_neighbours_of_a_plain_element():
    """Test that qubits given as one plain element on a lattice couple their neighbours as a chain of spins does.

    The Pauli matrices of a qubit are twice the spin matrices of a spin one half, so the coupling is four times.
    """
    states = ['up'] * 5 + ['down']
    plain = System(
        elements=[Element('chain', values={'psi': states, 'hilbert': QUBIT}, space=LatticeSpace(6))],
        interactions=[ExchangeInteraction('h', among='chain', J=0.7)],
    )
    spins = _chain(6, states, [ExchangeInteraction('h', among='chain', J=4 * 0.7)])
    problems = [QuantumCompiler().prepare(system).run().problem_ for system in (plain, spins)]
    matrices = [problem._get_generator_at(0.0, problem.initial_parameters).toarray() for problem in problems]
    assert np.allclose(matrices[0], matrices[1])
    assert np.array_equal(problems[0].initial_state, problems[1].initial_state)
    assert problems[0].dimensions == (2,) * 6


def test_compiler_reads_the_complex_value_of_a_plain_element_as_its_state():
    """Test that the state of a plain element is its value with complex allowed values, or its only other value."""
    assert OperatorCompiler._get_plain_state(TwoStateElement('q')) == 'phi'
    assert OperatorCompiler._get_plain_state(Element('q', values={'psi': 'up', 'hilbert': QUBIT})) == 'psi'


def test_system_runs_a_plain_quantum_element_with_the_compiler_its_hilbert_space_names():
    """Test that a system of a plain element that takes a Hilbert space prepares the quantum problem by itself."""
    system = System(elements=[Element('qubit', values={'psi': 'up', 'hilbert': QUBIT})]).prepare()
    assert type(system.solver_).__name__ == 'LindbladSolver'


def test_system_runs_a_one_off_interaction_of_a_qubit():
    """Test that a one-off interaction binds a qubit by its roles and gives its generator, as `x` flips it."""
    flip = Interaction(
        'flip',
        roles={'target': 'q'},
        laws=lambda s: [GeneratorLaw(2 * local_operator(s.target.psi, 'x'))],
    )
    system = System(elements=[QubitElement('q')], interactions=[flip]).prepare()
    observable = Observable('sz', local_z, group='q', reduce='sum')
    trajectory = system.simulate(until=math.pi / 4, dt=math.pi / 4, observables=[observable])
    assert trajectory.get_values('sz')[-1] == pytest.approx(-1.0, abs=1e-6)
