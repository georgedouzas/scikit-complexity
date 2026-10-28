"""Compile systems into problems of one mathematical form with fast numeric functions."""

from ._compilation import Compilation
from ._compile import compile_system, compiled_problem, partition_system
from ._issues import Issue, ValidationWarning
from ._kinetic_distributions import distribution_from, members_from
from ._lattice_steps import CompiledExtremal, ExtremalRecord
from ._observables import ObservableEvaluator, observables_of
from ._pipeline import BasePass, order_passes, run_passes
from ._quantum_operators import LocalOperator, Operator
from ._quantum_views import QuantumView
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
    'members_from',
    'observables_of',
    'order_passes',
    'partition_system',
    'run_passes',
    'validate',
]
