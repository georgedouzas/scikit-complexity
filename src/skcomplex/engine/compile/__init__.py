"""Compile systems into problems of one mathematical form with fast numeric functions."""

from ._base import BaseCompiler, BaseLowering, BasePass
from ._compilation import Compilation
from ._compile import compile_system
from ._compiled_extremal import CompiledExtremal
from ._extremal_record import ExtremalRecord
from ._issue import Issue
from ._kinetic_distributions import distribution_from, members_from
from ._local_operator import LocalOperator
from ._observables import ObservableEvaluator
from ._operator import Operator
from ._pass_compiler import PassCompiler
from ._patch_structure import PatchStructure
from ._quantum_views import QuantumView
from ._validation_report import ValidationReport
from ._validation_warning import ValidationWarning
from ._word_structure import WordStructure
from .passes._hilbert_space_error import HilbertSpaceError
from .passes._quantum_compiler import QuantumCompiler
from .passes._rewriting_compiler import RewritingCompiler

__all__: list[str] = [
    'BaseCompiler',
    'BaseLowering',
    'BasePass',
    'Compilation',
    'CompiledExtremal',
    'ExtremalRecord',
    'HilbertSpaceError',
    'Issue',
    'LocalOperator',
    'ObservableEvaluator',
    'Operator',
    'PassCompiler',
    'PatchStructure',
    'QuantumCompiler',
    'QuantumView',
    'RewritingCompiler',
    'ValidationReport',
    'ValidationWarning',
    'WordStructure',
    'compile_system',
    'distribution_from',
    'members_from',
]
