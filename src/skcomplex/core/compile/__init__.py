"""Compile systems into problems of one mathematical form with fast numeric functions."""

from ._arrays import Vector
from ._backends import BACKENDS, Backend, Lowering, reads_reductions, register_backend, resolve_backend
from ._capacity import CapacityRows, capacity_rows
from ._census import Census, CensusRecord, OwnerRule, register_owner
from ._compilation import BlockLaws, Compilation, Laws
from ._compile import compile_system, partition_system
from ._declarations import Block, Declarations, is_interaction, member_count
from ._forms import register_combined, register_form
from ._hybrid import CompiledEvent, EventRecord
from ._issues import Issue, ValidationWarning, names_of
from ._kernels import (
    LATTICE_BOUNDARIES,
    MODES,
    EdgeSearch,
    Kernel,
    Lowered,
    NeighborRead,
    NoiseProcess,
    Output,
    Reference,
    Source,
)
from ._layout import Entry, Layout
from ._observables import ObservableEvaluator, observables_of
from ._pipeline import Check, Pass, order_passes, register_check, register_pass, run_passes
from ._planner import SOLVERS, solver_classes, solver_registry
from ._printing import NUMPY_DIALECT, SCALAR_DIALECT, Dialect, numpy_function, print_expressions
from ._problem import (
    Accelerated,
    CompiledAgents,
    CompiledElement,
    CompiledInteraction,
    Draws,
    FractionalRows,
    Getter,
    NetworkSnapshot,
    Partition,
    Problem,
    Record,
    RenewalRecord,
    VectorFunction,
    with_initial_state,
)
from ._quantities import LoopTerm, StockFlow
from ._reductions import ReductionPass
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
from ._system_passes import SystemPass, claiming_pass
from ._tree import Node, coerce_value, declared_times, flatten_system
from ._validation import ValidationReport, missing_declarations, validate
from .passes._problem import visual_of

__all__: list[str] = [
    'BACKENDS',
    'LATTICE_BOUNDARIES',
    'MODES',
    'NUMPY_DIALECT',
    'SCALAR_DIALECT',
    'SOLVERS',
    'Accelerated',
    'Backend',
    'Binding',
    'Block',
    'BlockLaws',
    'CapacityRows',
    'Census',
    'CensusRecord',
    'Check',
    'Compilation',
    'CompiledAgents',
    'CompiledElement',
    'CompiledEvent',
    'CompiledInteraction',
    'Declarations',
    'Dialect',
    'Draws',
    'EdgeSearch',
    'Entry',
    'EventRecord',
    'FractionalRows',
    'Getter',
    'Issue',
    'Kernel',
    'KernelSpec',
    'Laws',
    'Layout',
    'LoopTerm',
    'Lowered',
    'Lowering',
    'NeighborRead',
    'NetworkSnapshot',
    'Node',
    'NoiseProcess',
    'ObservableEvaluator',
    'Output',
    'OwnerRule',
    'Partition',
    'Pass',
    'Problem',
    'Record',
    'ReductionPass',
    'Reference',
    'RenewalRecord',
    'Source',
    'StockFlow',
    'SystemPass',
    'ValidationReport',
    'ValidationWarning',
    'Vector',
    'VectorFunction',
    'capacity_rows',
    'claiming_pass',
    'coerce_value',
    'compile_system',
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
    'numpy_function',
    'observables_of',
    'order_passes',
    'partition_system',
    'print_expressions',
    'reads_reductions',
    'register_backend',
    'register_binding',
    'register_check',
    'register_combined',
    'register_form',
    'register_owner',
    'register_pass',
    'resolve_backend',
    'run_passes',
    'solver_classes',
    'solver_registry',
    'spectral_multiplier',
    'state_name',
    'target_references',
    'validate',
    'visual_of',
    'with_initial_state',
]
