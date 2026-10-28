"""Compile systems into problems of one mathematical form with fast numeric functions."""

from ._compilation import Compilation
from ._compile import compile_system, compiled_problem, partition_system
from ._compiled_extremal import CompiledExtremal
from ._extremal_record import ExtremalRecord
from ._issue import Issue
from ._local_operator import LocalOperator
from ._observables import ObservableEvaluator, observables_of
from ._operator import Operator
from ._patch_structure import PatchStructure
from ._pipeline import BasePass, order_passes, run_passes
from ._quantum_views import QuantumView
from ._system_compilers import BaseSystemCompiler
from ._validation import ValidationReport, validate
from ._validation_warning import ValidationWarning
from ._word_structure import WordStructure
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
    'observables_of',
    'order_passes',
    'partition_system',
    'run_passes',
    'validate',
]
