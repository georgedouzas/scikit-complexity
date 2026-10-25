"""Compile systems into problems of one mathematical form with fast numeric functions."""

from ._capacity import CapacityRows, capacity_rows
from ._census import Census, CensusRecord, OwnerRule, register_owner
from ._compilation import BlockLaws, Compilation, Laws
from ._compile import compile_system, compiled_problem, partition_system
from ._declarations import Block, Declarations, is_interaction, member_count
from ._forms import register_combined, register_form
from ._issues import Issue, ValidationWarning, names_of
from ._observables import ObservableEvaluator, observables_of
from ._pipeline import BasePass, Check, order_passes, register_check, register_pass, run_passes
from ._planner import SOLVERS, solver_classes, solver_registry
from ._specs import (
    Binding,
    KernelSpec,
    group_space,
    mask_of,
    masked_value,
    member_spec,
    register_binding,
    state_name,
    target_references,
)
from ._spectral import spectral_multiplier
from ._system_compilers import BaseSystemCompiler, claiming_compiler, register_system_compiler
from ._tree import Node, coerce_value, declared_times, flatten_system
from ._validation import ValidationReport, missing_declarations, validate
from .passes._problem import visual_of

__all__: list[str] = [
    'SOLVERS',
    'BasePass',
    'BaseSystemCompiler',
    'Binding',
    'Block',
    'BlockLaws',
    'CapacityRows',
    'Census',
    'CensusRecord',
    'Check',
    'Compilation',
    'Declarations',
    'Issue',
    'KernelSpec',
    'Laws',
    'Node',
    'ObservableEvaluator',
    'OwnerRule',
    'ValidationReport',
    'ValidationWarning',
    'capacity_rows',
    'claiming_compiler',
    'coerce_value',
    'compile_system',
    'compiled_problem',
    'declared_times',
    'flatten_system',
    'group_space',
    'is_interaction',
    'mask_of',
    'masked_value',
    'member_count',
    'member_spec',
    'missing_declarations',
    'names_of',
    'observables_of',
    'order_passes',
    'partition_system',
    'register_binding',
    'register_check',
    'register_combined',
    'register_form',
    'register_owner',
    'register_pass',
    'register_system_compiler',
    'run_passes',
    'solver_classes',
    'solver_registry',
    'spectral_multiplier',
    'state_name',
    'target_references',
    'validate',
    'visual_of',
]
