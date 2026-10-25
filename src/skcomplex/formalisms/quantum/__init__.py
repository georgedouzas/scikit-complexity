"""Quantum dynamics: operators, Hamiltonian and dissipator laws, quantum elements, and QuTiP solvers."""

from ...core.base import Form
from .._formalism import Formalism, register
from ._components import HilbertDomain, HilbertFamilyDomain, QuantumView, hilbert_state, is_quantum, view_of
from ._laws import DissipatorLaw, GeneratorLaw, LocalOperator, Operator, sparse_matrix
from ._observables import expectation
from ._passes import (
    DEFAULT_MAX_DIMENSION,
    HilbertSpaceError,
    QuantumCompiler,
    QuantumProblem,
    QuantumTerm,
    contains_quantum,
)
from ._solvers import LindbladSolver, QuantumSolver, QuantumTrajectoriesSolver, UnitarySolver
from ._visuals import ExpectationVisual

__all__: list[str] = [
    'DEFAULT_MAX_DIMENSION',
    'DissipatorLaw',
    'ExpectationVisual',
    'GeneratorLaw',
    'HilbertDomain',
    'HilbertFamilyDomain',
    'HilbertSpaceError',
    'LindbladSolver',
    'LocalOperator',
    'Operator',
    'QuantumCompiler',
    'QuantumProblem',
    'QuantumSolver',
    'QuantumTerm',
    'QuantumTrajectoriesSolver',
    'QuantumView',
    'UnitarySolver',
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
        laws=(GeneratorLaw, DissipatorLaw),
        compilers=(QuantumCompiler(),),
        forms=(Form.QUANTUM,),
        solvers={
            'unitary': UnitarySolver,
            'lindblad': LindbladSolver,
            'quantum_trajectories': QuantumTrajectoriesSolver,
        },
        defaults=(LindbladSolver,),
        visuals={'expectations': ExpectationVisual},
    ),
)
