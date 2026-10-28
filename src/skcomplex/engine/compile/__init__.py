"""Compile systems into problems of one mathematical form with fast numeric functions."""

from ._compilation import Compilation
from ._compile import compile_system, compiled_problem, partition_system
from ._field_lowering import mesh_laplacian
from ._issues import Issue, ValidationWarning
from ._kinetic_distributions import distribution_from, members_from
from ._lattice_steps import CompiledExtremal, ExtremalRecord
from ._observables import ObservableEvaluator, held_reductions_of, namespace_of, observables_of
from ._pipeline import BasePass, order_passes, run_passes
from ._probe import runs_rules
from ._quantum_operators import LocalOperator, Operator
from ._quantum_views import QuantumView
from ._spectral import spectral_multiplier
from ._structures import PatchStructure, WordStructure
from ._system_compilers import BaseSystemCompiler
from ._validation import ValidationReport, validate
from .passes._core import claiming_compiler

__all__: list[str] = [
    'BasePass',
    'BaseSystemCompiler',
    'Compilation',
    'CompiledExtremal',
    'ExtremalRecord',
    'Issue',
    'LocalOperator',
    'ObservableEvaluator',
    'Operator',
    'PatchStructure',
    'QuantumView',
    'ValidationReport',
    'ValidationWarning',
    'WordStructure',
    'claiming_compiler',
    'compile_system',
    'compiled_problem',
    'distribution_from',
    'held_reductions_of',
    'members_from',
    'mesh_laplacian',
    'namespace_of',
    'observables_of',
    'order_passes',
    'partition_system',
    'run_passes',
    'runs_rules',
    'spectral_multiplier',
    'validate',
]
