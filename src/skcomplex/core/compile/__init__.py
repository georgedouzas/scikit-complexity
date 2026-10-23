"""Compile systems into problems of one mathematical form with fast numeric functions."""

from ._backends import BACKENDS, Backend, Lowering, load_backends, register_backend, resolve_backend
from ._census import Census, CensusRecord, OwnerRule, law_census, register_owner
from ._compilation import Compilation, Laws
from ._compile import compile_system, partition_system
from ._declarations import Declarations, member_count
from ._fields import mesh_laplacian, spectral_multiplier
from ._forms import register_combined, register_form
from ._hybrid import Breakpoints, CompiledEvent, Delay, EventRecord
from ._issues import Issue, ValidationWarning
from ._kernels import Reference, Source
from ._lattices import (
    CompiledDrive,
    CompiledExtremal,
    CompiledMonteCarlo,
    CompiledRelaxation,
    CompiledRule,
    ExtremalRecord,
    LatticeState,
    LatticeStep,
    RelaxationRecord,
)
from ._layout import Entry, Layout
from ._observables import ObservableEvaluator, observables_of, with_observables
from ._pipeline import Check, Pass, order_passes, register_check, register_pass, registered_passes, run_passes
from ._planner import Plan, PlannedPartition, plan_partitions
from ._printing import NUMPY_DIALECT, SCALAR_DIALECT, Dialect, numpy_function, print_expressions
from ._problem import (
    Accelerated,
    CompiledAgents,
    CompiledElement,
    CompiledInteraction,
    Draws,
    FractionalRows,
    Getter,
    JumpChannels,
    MarkovMatrix,
    NetworkSnapshot,
    Partition,
    Problem,
    RenewalProcess,
    RenewalRecord,
    TopologyChanges,
    Vector,
    VectorFunction,
    with_initial_state,
)
from ._quantities import CompiledFlow, CompiledObjective, LoopTerm, Recorded, StockFlow
from ._specs import Binding, register_binding
from ._system_passes import SystemPass, claiming_pass
from ._tree import Node, coerce_value, flatten_system
from ._validation import ValidationReport, domain_kind, missing_declarations, validate

__all__: list[str] = [
    'BACKENDS',
    'NUMPY_DIALECT',
    'SCALAR_DIALECT',
    'Accelerated',
    'Backend',
    'Binding',
    'Breakpoints',
    'Census',
    'CensusRecord',
    'Check',
    'Compilation',
    'CompiledAgents',
    'CompiledDrive',
    'CompiledElement',
    'CompiledEvent',
    'CompiledExtremal',
    'CompiledFlow',
    'CompiledInteraction',
    'CompiledMonteCarlo',
    'CompiledObjective',
    'CompiledRelaxation',
    'CompiledRule',
    'Declarations',
    'Delay',
    'Dialect',
    'Draws',
    'Entry',
    'EventRecord',
    'ExtremalRecord',
    'FractionalRows',
    'Getter',
    'Issue',
    'JumpChannels',
    'LatticeState',
    'LatticeStep',
    'Laws',
    'Layout',
    'LoopTerm',
    'Lowering',
    'MarkovMatrix',
    'NetworkSnapshot',
    'Node',
    'ObservableEvaluator',
    'OwnerRule',
    'Partition',
    'Pass',
    'Plan',
    'PlannedPartition',
    'Problem',
    'Recorded',
    'Reference',
    'RelaxationRecord',
    'RenewalProcess',
    'RenewalRecord',
    'Source',
    'StockFlow',
    'SystemPass',
    'TopologyChanges',
    'ValidationReport',
    'ValidationWarning',
    'Vector',
    'VectorFunction',
    'claiming_pass',
    'coerce_value',
    'compile_system',
    'domain_kind',
    'flatten_system',
    'law_census',
    'load_backends',
    'member_count',
    'mesh_laplacian',
    'missing_declarations',
    'numpy_function',
    'observables_of',
    'order_passes',
    'partition_system',
    'plan_partitions',
    'print_expressions',
    'register_backend',
    'register_binding',
    'register_check',
    'register_combined',
    'register_form',
    'register_owner',
    'register_pass',
    'registered_passes',
    'resolve_backend',
    'run_passes',
    'spectral_multiplier',
    'validate',
    'with_initial_state',
    'with_observables',
]
