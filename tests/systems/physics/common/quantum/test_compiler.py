"""Test the `_compiler` module of `skcomplex.systems.physics.common.quantum`."""

import numpy as np
import pytest

from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import local_operator
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine._utils import symbol
from skcomplex.core.engine.compile import OperatorCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import IncompatibleLawsError
from skcomplex.systems.physics.common.quantum import (
    DissipatorLaw,
    GeneratorLaw,
    LindbladSolver,
    QuantumCompiler,
    QubitElement,
    UnitarySolver,
)


class DrivenElement(QubitElement):
    """Qubit with a field along x and a decay."""

    def _get_laws(self, s):
        """Return the field and the decay."""
        return [GeneratorLaw(local_operator(s.psi, 'x')), DissipatorLaw(local_operator(s.psi, 'minus'), 0.25)]


class ClassicalElement(QubitElement):
    """Qubit with a classical rate law."""

    def _get_laws(self, s):
        """Return the classical rate law."""
        return [RateLaw(symbol('x'), 1)]


def test_quantum_compiler_adds_generators_and_dissipators():
    """Test that the compiler puts the generator law in the Hamiltonian and the dissipator law in the dissipators."""
    compiler = QuantumCompiler().prepare(CompositeSystem(elements=[DrivenElement('q')])).run()
    problem = compiler.problem_
    assert issubclass(QuantumCompiler, OperatorCompiler)
    assert np.allclose(problem._get_generator_at(0.0, problem.initial_parameters).toarray(), [[0, 1], [1, 0]])
    assert len(problem.dissipators) == 1
    assert not problem.pure


def test_quantum_compiler_names_the_lindblad_solver_unless_given_one():
    """Test that the compiler names `LindbladSolver` by default and keeps a solver it is given."""
    system = CompositeSystem(elements=[QubitElement('q')])
    assert isinstance(QuantumCompiler().prepare(system).run().solver_, LindbladSolver)
    assert isinstance(QuantumCompiler(solver=UnitarySolver()).prepare(system).run().solver_, UnitarySolver)


def test_quantum_compiler_rejects_classical_laws():
    """Test that a law other than a generator or a dissipator raises an error that names it."""
    with pytest.raises(IncompatibleLawsError, match='RateLaw, which QuantumCompiler cannot compile'):
        QuantumCompiler().prepare(CompositeSystem(elements=[ClassicalElement('q')])).run()


def test_systems_of_quantum_elements_compile_with_the_quantum_compiler():
    """Test that a simulation of quantum elements picks the compiler their Hilbert space names."""
    trajectory = simulate(CompositeSystem(elements=[DrivenElement('q')]), until=1.0, dt=1.0)
    assert trajectory.get_values('purity')[-1] < 1
