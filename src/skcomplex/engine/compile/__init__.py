"""Compile systems into problems of one mathematical form with fast numeric functions."""

from ._capacity import CapacityRows, capacity_rows
from ._census import Census, CensusRecord
from ._compilation import BlockLaws, Compilation, Laws
from ._compile import compile_system, compiled_problem, partition_system
from ._declarations import Block, Declarations, is_interaction, member_count
from ._field_lowering import mesh_laplacian
from ._forms import COMBINED, FORMS, STEPWISE_FORMS
from ._issues import Issue, ValidationWarning, names_of
from ._kinetic_contractions import CompiledEncounter, TableValues
from ._kinetic_distributions import distribution_from, members_from
from ._lattice_steps import (
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
from ._laws import replaced, values_of
from ._observables import ObservableEvaluator, held_reductions_of, namespace_of, observables_of
from ._pipeline import BasePass, Check, order_passes, run_passes
from ._probe import runs_rules
from ._quantum_operators import LocalOperator, Operator, sparse_matrix
from ._quantum_views import HilbertDomain, HilbertFamilyDomain, QuantumView, hilbert_state, is_quantum, view_of
from ._renewal_law import RenewalLaw
from ._rewriting_words import CompiledWord
from ._specs import (
    KernelSpec,
    group_space,
    mask_of,
    masked_value,
    member_spec,
    state_name,
    target_references,
)
from ._spectral import spectral_multiplier
from ._structures import GraphStructure, PatchStructure, Structure, WordStructure, affine_matrix
from ._system_compilers import BaseSystemCompiler
from ._topology import CompiledRequest, NetworkRows, Topology, request_of
from ._tree import Node, coerce_value, declared_times, flatten_system
from ._validation import ValidationReport, validate
from .passes._core import CHECKS, PASSES, SYSTEM_COMPILERS, claiming_compiler
from .passes._problem import visual_of

__all__: list[str] = [
    'CHECKS',
    'COMBINED',
    'FORMS',
    'PASSES',
    'STEPWISE_FORMS',
    'SYSTEM_COMPILERS',
    'BasePass',
    'BaseSystemCompiler',
    'Block',
    'BlockLaws',
    'CapacityRows',
    'Census',
    'CensusRecord',
    'Check',
    'Compilation',
    'CompiledDrive',
    'CompiledEncounter',
    'CompiledExtremal',
    'CompiledMonteCarlo',
    'CompiledRelaxation',
    'CompiledRequest',
    'CompiledRule',
    'CompiledWord',
    'Declarations',
    'ExtremalRecord',
    'GraphStructure',
    'HilbertDomain',
    'HilbertFamilyDomain',
    'Issue',
    'KernelSpec',
    'LatticeState',
    'LatticeStep',
    'Laws',
    'LocalOperator',
    'NetworkRows',
    'Node',
    'ObservableEvaluator',
    'Operator',
    'PatchStructure',
    'QuantumView',
    'RelaxationRecord',
    'RenewalLaw',
    'Structure',
    'TableValues',
    'Topology',
    'ValidationReport',
    'ValidationWarning',
    'WordStructure',
    'affine_matrix',
    'capacity_rows',
    'claiming_compiler',
    'coerce_value',
    'compile_system',
    'compiled_problem',
    'declared_times',
    'distribution_from',
    'flatten_system',
    'group_space',
    'held_reductions_of',
    'hilbert_state',
    'is_interaction',
    'is_quantum',
    'mask_of',
    'masked_value',
    'member_count',
    'member_spec',
    'members_from',
    'mesh_laplacian',
    'names_of',
    'namespace_of',
    'observables_of',
    'order_passes',
    'partition_system',
    'replaced',
    'request_of',
    'run_passes',
    'runs_rules',
    'sparse_matrix',
    'spectral_multiplier',
    'state_name',
    'target_references',
    'validate',
    'values_of',
    'view_of',
    'visual_of',
]
