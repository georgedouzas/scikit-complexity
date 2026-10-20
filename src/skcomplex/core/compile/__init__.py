"""Compile systems into problems of one mathematical form with fast numeric functions."""

from ._backends import BACKENDS, Backend, Lowering, NumpyBackend, register_backend, resolve_backend
from ._compiler import compile_system, partition_system
from ._fields import mesh_laplacian, spectral_multiplier
from ._hybrid import Breakpoints, CompiledEvent, Delay, EventRecord
from ._lattices import (
    CompiledDrive,
    CompiledMonteCarlo,
    CompiledRelaxation,
    CompiledRule,
    LatticeState,
    LatticeStep,
    RelaxationRecord,
)
from ._layout import Entry, Layout
from ._numba_backend import NumbaBackend
from ._observables import ObservableEvaluator, observables_of
from ._printing import NUMPY_DIALECT, SCALAR_DIALECT, Dialect, numpy_function, print_expressions
from ._problem import (
    Accelerated,
    CompiledAgents,
    CompiledElement,
    CompiledInteraction,
    Draws,
    JumpChannels,
    MarkovMatrix,
    Partition,
    Problem,
    Vector,
    VectorFunction,
)

__all__: list[str] = [
    'BACKENDS',
    'NUMPY_DIALECT',
    'SCALAR_DIALECT',
    'Accelerated',
    'Backend',
    'Breakpoints',
    'CompiledAgents',
    'CompiledDrive',
    'CompiledElement',
    'CompiledEvent',
    'CompiledInteraction',
    'CompiledMonteCarlo',
    'CompiledRelaxation',
    'CompiledRule',
    'Delay',
    'Dialect',
    'Draws',
    'Entry',
    'EventRecord',
    'JumpChannels',
    'LatticeState',
    'LatticeStep',
    'Layout',
    'Lowering',
    'MarkovMatrix',
    'NumbaBackend',
    'NumpyBackend',
    'ObservableEvaluator',
    'Partition',
    'Problem',
    'RelaxationRecord',
    'Vector',
    'VectorFunction',
    'compile_system',
    'mesh_laplacian',
    'numpy_function',
    'observables_of',
    'partition_system',
    'print_expressions',
    'register_backend',
    'resolve_backend',
    'spectral_multiplier',
]
