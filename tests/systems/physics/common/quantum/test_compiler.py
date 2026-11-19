"""Test the `_compiler` module of `skcomplex.systems.physics.common.quantum`."""

from types import SimpleNamespace

import numpy as np
import pytest

from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import local_operator
from skcomplex.core.components.spaces import BaseSpace
from skcomplex.core.engine._utils import symbol
from skcomplex.core.engine.compile import OperatorCompiler
from skcomplex.core.errors import IncompatibleLawsError
from skcomplex.core.specs import Role
from skcomplex.core.systems import System
from skcomplex.systems.physics.common.quantum import (
    DissipatorLaw,
    GeneratorLaw,
    LindbladSolver,
    QuantumCompiler,
    QubitElement,
    UnitarySolver,
)


class DriveInteraction(Interaction):
    """Field along x and decay of a qubit."""

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the field and the decay."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            psi = context.target.psi
            self.laws_ = [GeneratorLaw(local_operator(psi, 'x')), DissipatorLaw(local_operator(psi, 'minus'), 0.25)]


class ClassicalInteraction(Interaction):
    """Classical rate law on a qubit."""

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the classical rate law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [RateLaw(symbol('x'), 1)]


def _qubit(*interactions: Interaction) -> System:
    """Return a qubit under some interactions."""
    return System(elements=[QubitElement('q')], interactions=list(interactions))


def test_quantum_compiler_adds_generators_and_dissipators():
    """Test that the compiler puts the generator law in the Hamiltonian and the dissipator law in the dissipators."""
    compiler = QuantumCompiler().prepare(_qubit(DriveInteraction('drive', target='q'))).run()
    problem = compiler.problem_
    assert issubclass(QuantumCompiler, OperatorCompiler)
    assert np.allclose(problem._get_generator_at(0.0, problem.initial_parameters).toarray(), [[0, 1], [1, 0]])
    assert len(problem.dissipators) == 1
    assert not problem.pure


def test_quantum_compiler_names_the_lindblad_solver_unless_given_one():
    """Test that the compiler names `LindbladSolver` by default and keeps a solver it is given."""
    system = _qubit()
    assert isinstance(QuantumCompiler().prepare(system).run().solver_, LindbladSolver)
    assert isinstance(QuantumCompiler(solver=UnitarySolver()).prepare(system).run().solver_, UnitarySolver)


def test_quantum_compiler_rejects_classical_laws():
    """Test that a law other than a generator or a dissipator raises an error that names it."""
    with pytest.raises(IncompatibleLawsError, match='RateLaw, which QuantumCompiler cannot compile'):
        QuantumCompiler().prepare(_qubit(ClassicalInteraction('classical', target='q'))).run()


def test_systems_of_quantum_elements_compile_with_the_quantum_compiler():
    """Test that a simulation of quantum elements picks the compiler their Hilbert space names."""
    trajectory = _qubit(DriveInteraction('drive', target='q')).prepare().simulate(until=1.0, dt=1.0)
    assert trajectory.get_values('purity')[-1] < 1
