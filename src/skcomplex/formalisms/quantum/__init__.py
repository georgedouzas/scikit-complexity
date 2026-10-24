"""Quantum dynamics: operators, Hamiltonian and dissipator laws, quantum elements, and QuTiP solvers."""

from ...core.base import Form
from .._formalism import Formalism, register
from ._components import QuantumView, hilbert_state, is_quantum, view_of
from ._laws import Dissipator, Generator, LocalOperator, Operator, sparse_matrix
from ._observables import expectation
from ._passes import DEFAULT_MAX_DIMENSION, QuantumPass, QuantumProblem, QuantumTerm, contains_quantum
from ._solvers import LindbladEvolution, QuantumSolver, QuantumTrajectories, UnitaryEvolution
from ._visuals import ExpectationVisual

__all__: list[str] = [
    'DEFAULT_MAX_DIMENSION',
    'Dissipator',
    'ExpectationVisual',
    'Generator',
    'LindbladEvolution',
    'LocalOperator',
    'Operator',
    'QuantumPass',
    'QuantumProblem',
    'QuantumSolver',
    'QuantumTerm',
    'QuantumTrajectories',
    'QuantumView',
    'UnitaryEvolution',
    'contains_quantum',
    'expectation',
    'hilbert_state',
    'is_quantum',
    'sparse_matrix',
    'view_of',
]

register(
    Formalism(
        'quantum',
        laws=(Generator, Dissipator),
        passes=(QuantumPass(),),
        forms=(Form.QUANTUM,),
        solvers={
            'unitary': UnitaryEvolution,
            'lindblad': LindbladEvolution,
            'quantum_trajectories': QuantumTrajectories,
        },
        defaults=(LindbladEvolution,),
        visuals={'expectations': ExpectationVisual},
    ),
)
