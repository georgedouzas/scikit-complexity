"""Test the `OperatorCompiler` class."""

from operator import attrgetter
from typing import Annotated

import numpy as np
import pytest
import qutip
from sage.all__sagemath_symbolics import CC

from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.interactions import NeighborInteraction
from skcomplex.core.components.laws import DissipatorLaw, GeneratorLaw, RateLaw
from skcomplex.core.components.laws.utils import local_operator
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine._utils import symbol
from skcomplex.core.engine.compile import BaseCompiler, BasePass
from skcomplex.core.engine.compile._observables import ObservableEvaluator
from skcomplex.core.engine.compile.passes._operator_compiler import OperatorCompiler
from skcomplex.core.errors import IncompatibleLawsError, ValidationError
from skcomplex.core.specs import HilbertSpec, ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from skcomplex.systems.biology.common.populations import PopulationElement
from skcomplex.systems.physics.common.quantum import QUBIT, QubitElement

from .....conftest import SpinElement, _spin_space


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
    """Qubit with a constant generator."""

    def _get_laws(self, s):
        """Return the constant generator."""
        return [GeneratorLaw(0.5 * local_operator(s.psi, 'z') + local_operator(s.psi, 'x'))]


class PlusElement(QubitElement):
    """Qubit whose generator is a ladder operator without its adjoint."""

    def _get_laws(self, s):
        """Return the non-Hermitian generator."""
        return [GeneratorLaw(local_operator(s.psi, 'plus'))]


class ForeignElement(QubitElement):
    """Qubit whose generator reads a classical symbol."""

    def _get_laws(self, s):
        """Return the generator with a classical coefficient."""
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


def test_compiler_counts_one_exchange_term_per_bond():
    """Test that an exchange on a periodic lattice of 8 spins equals the sum of the 8 bond terms."""
    system = CompositeSystem(
        elements=[
            GroupElement(
                'chain',
                element=SpinElement,
                space=LatticeSpace(8),
                values={'hilbert': _spin_space()},
                on='sites',
            ),
        ],
        interactions=[HeisenbergInteraction('h', target='chain')],
    )
    problem = OperatorCompiler().prepare(system).run().problem_
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
    system = CompositeSystem(
        elements=[
            FixedElement('q'),
        ],
    )
    problem = OperatorCompiler().prepare(system).run().problem_
    assert len(problem.generator) == 1
    assert problem.generator[0].function is None


def test_compiler_rejects_non_hermitian_generators():
    """Test that a generator without its adjoint raises an error that names the law."""
    with pytest.raises(ValidationError, match=r'generator of q is not Hermitian.*adjoint'):
        OperatorCompiler().prepare(CompositeSystem(elements=[PlusElement('q')])).run()


def test_compiler_rejects_classical_symbols_in_coefficients():
    """Test that a coefficient reading a symbol other than parameters and time raises an error."""
    with pytest.raises(ValidationError, match='n_rabbits'):
        OperatorCompiler().prepare(CompositeSystem(elements=[ForeignElement('q')])).run()


def test_compiler_rejects_classical_elements_and_laws():
    """Test that operator systems hold neither classical elements nor classical laws."""
    with pytest.raises(IncompatibleLawsError, match='rabbits'):
        OperatorCompiler().prepare(CompositeSystem(elements=[QubitElement('q'), PopulationElement('rabbits')])).run()
    with pytest.raises(IncompatibleLawsError, match='RateLaw'):
        OperatorCompiler().prepare(CompositeSystem(elements=[ClassicalElement('q')])).run()


def test_compiler_rejects_negative_rates():
    """Test that a dissipator with a negative rate raises an error."""
    with pytest.raises(ValidationError, match='non-negative'):
        OperatorCompiler().prepare(CompositeSystem(elements=[NegativeElement('q')])).run()


def test_compiler_stores_neel_states():
    """Test that a Néel state has norm 1 and one nonzero amplitude at its basis index."""
    labels = ['up', 'down'] * 3
    system = CompositeSystem(
        elements=[
            GroupElement('chain', element=SpinElement, space=LatticeSpace(6), values={'psi': labels}, on='sites'),
        ],
    )
    problem = OperatorCompiler().prepare(system).run().problem_
    state = problem._get_state_of(problem.initial_state)
    assert np.linalg.norm(state) == pytest.approx(1)
    assert np.flatnonzero(state).tolist() == [int('010101', 2)]


@pytest.mark.parametrize(('initial', 'message'), [(np.array([2.0, 0.0]), 'norm 2'), ('hot', 'ground')])
def test_compiler_rejects_invalid_initial_states(initial, message):
    """Test that an unnormalised vector or an unknown name raises an error that names the system."""
    with pytest.raises(ValidationError, match=message):
        OperatorCompiler().prepare(CompositeSystem('box', elements=[QubitElement('q')])).run({'box.state': initial})


def test_compiler_stores_density_matrices_of_open_systems():
    """Test that an open system and a mixed initial state compile to density matrices."""
    problem = OperatorCompiler().prepare(CompositeSystem(elements=[DecayingElement('q')])).run().problem_
    assert not problem.pure
    assert problem._get_state_of(problem.initial_state).shape == (2, 2)
    system = CompositeSystem(elements=[QubitElement('q')])
    mixed = OperatorCompiler().prepare(system).run({'system.state': np.eye(2) / 2}).problem_
    assert not mixed.pure


def test_compiler_starts_from_the_ground_state():
    """Test that the initial state 'ground' is the lowest eigenstate."""
    system = CompositeSystem(
        elements=[
            GroupElement(
                'chain',
                element=SpinElement,
                space=LatticeSpace(4),
                values={'hilbert': _spin_space()},
                on='sites',
            ),
        ],
        interactions=[HeisenbergInteraction('h', target='chain')],
    )
    problem = OperatorCompiler().prepare(system).run({'system.state': 'ground'}).problem_
    energy = (
        ObservableEvaluator(problem)
        .prepare()
        .evaluate(0.0, problem.initial_state, problem.initial_parameters)['energy']
    )
    assert energy == pytest.approx(-2.0)


def test_observable_evaluator_reads_local_magnetization():
    """Test that a group observable gives one expectation of the local z per site, and their sum."""
    labels = ['up', 'down'] * 4
    system = CompositeSystem(
        elements=[
            GroupElement(
                'chain',
                element=SpinElement,
                space=LatticeSpace(8),
                values={'psi': labels, 'hilbert': _spin_space()},
                on='sites',
            ),
        ],
    )
    observables = [
        ObservableSpec('sz', local_z, group='chain', reduce=np.asarray),
        ObservableSpec('total', local_z, group='chain', reduce='sum'),
    ]
    problem = OperatorCompiler(observables=observables).prepare(system).run().problem_
    values = ObservableEvaluator(problem).prepare().evaluate(0.0, problem.initial_state, problem.initial_parameters)
    assert values['sz'].tolist() == [0.5, -0.5] * 4
    assert values['total'] == 0.0


def test_observable_evaluator_rejects_non_operators():
    """Test that an operator observable must give an operator."""
    system = CompositeSystem(elements=[QubitElement('q')])
    with pytest.raises(ValidationError, match='must give an operator'):
        ObservableEvaluator(
            OperatorCompiler(observables=[ObservableSpec('label', attrgetter('t'))]).prepare(system).run().problem_,
        ).prepare()


def test_operator_compiler_reports_a_classical_system():
    """Test that the operator compiler accepts an operator system and reports a classical one."""
    compiler = OperatorCompiler()
    assert compiler.prepare(CompositeSystem(elements=[GroupElement('chain', element=SpinElement, size=2)])).report_.ok
    assert not compiler.prepare(CompositeSystem(elements=[PopulationElement('rabbits', x=10)])).report_.ok


def test_operator_compiler_is_a_system_compiler_and_not_a_pass():
    """Test that the operator compiler compiles whole systems and is not a step of the pipeline of passes."""
    assert issubclass(OperatorCompiler, BaseCompiler)
    assert not issubclass(OperatorCompiler, BasePass)
