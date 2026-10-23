"""Compile systems into problems of one mathematical form with fast numeric functions."""

from ._backends import BACKENDS, Backend, Lowering, load_backends, register_backend, resolve_backend
from ._census import Census, CensusRecord, law_census
from ._compilation import Compilation
from ._compile import compile_system, partition_system
from ._contractions import CompiledEncounter, TableValues
from ._fields import mesh_laplacian, spectral_multiplier
from ._hybrid import Breakpoints, CompiledEvent, Delay, EventRecord
from ._issues import Issue, ValidationWarning
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
from ._quantum import DEFAULT_MAX_DIMENSION, QuantumProblem, QuantumTerm, contains_quantum
from ._rewriting import CompiledGraph, CompiledPatch, compile_rewriting
from ._structures import GraphStructure, PatchStructure, Structure, WordStructure
from ._system_passes import SystemPass, claiming_pass
from ._validation import ValidationReport, domain_kind, missing_declarations, validate
from ._words import CompiledWord

__all__: list[str] = [
    'BACKENDS',
    'DEFAULT_MAX_DIMENSION',
    'NUMPY_DIALECT',
    'SCALAR_DIALECT',
    'Accelerated',
    'Backend',
    'Breakpoints',
    'Census',
    'CensusRecord',
    'Check',
    'Compilation',
    'CompiledAgents',
    'CompiledDrive',
    'CompiledElement',
    'CompiledEncounter',
    'CompiledEvent',
    'CompiledExtremal',
    'CompiledFlow',
    'CompiledGraph',
    'CompiledInteraction',
    'CompiledMonteCarlo',
    'CompiledObjective',
    'CompiledPatch',
    'CompiledRelaxation',
    'CompiledRule',
    'CompiledWord',
    'Delay',
    'Dialect',
    'Draws',
    'Entry',
    'EventRecord',
    'ExtremalRecord',
    'FractionalRows',
    'GraphStructure',
    'Issue',
    'JumpChannels',
    'LatticeState',
    'LatticeStep',
    'Layout',
    'LoopTerm',
    'Lowering',
    'MarkovMatrix',
    'NetworkSnapshot',
    'ObservableEvaluator',
    'Partition',
    'Pass',
    'PatchStructure',
    'Plan',
    'PlannedPartition',
    'Problem',
    'QuantumProblem',
    'QuantumTerm',
    'Recorded',
    'RelaxationRecord',
    'RenewalProcess',
    'RenewalRecord',
    'StockFlow',
    'Structure',
    'SystemPass',
    'TableValues',
    'TopologyChanges',
    'ValidationReport',
    'ValidationWarning',
    'Vector',
    'VectorFunction',
    'WordStructure',
    'claiming_pass',
    'compile_rewriting',
    'compile_system',
    'contains_quantum',
    'domain_kind',
    'law_census',
    'load_backends',
    'mesh_laplacian',
    'missing_declarations',
    'numpy_function',
    'observables_of',
    'order_passes',
    'partition_system',
    'plan_partitions',
    'print_expressions',
    'register_backend',
    'register_check',
    'register_pass',
    'registered_passes',
    'resolve_backend',
    'run_passes',
    'spectral_multiplier',
    'validate',
    'with_initial_state',
    'with_observables',
]
