"""Compile systems into problems of one mathematical form with fast numeric functions."""

from ._backends import BACKENDS, Backend, Lowering, NumpyBackend, register_backend, resolve_backend
from ._compiler import compile_system, partition_system
from ._layout import Entry, Layout
from ._numba_backend import NumbaBackend
from ._printing import NUMPY_DIALECT, SCALAR_DIALECT, Dialect, numpy_function, print_expressions
from ._problem import (
    Accelerated,
    CompiledAgents,
    CompiledElement,
    CompiledInteraction,
    JumpChannels,
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
    'CompiledAgents',
    'CompiledElement',
    'CompiledInteraction',
    'Dialect',
    'Entry',
    'JumpChannels',
    'Layout',
    'Lowering',
    'NumbaBackend',
    'NumpyBackend',
    'Partition',
    'Problem',
    'Vector',
    'VectorFunction',
    'compile_system',
    'numpy_function',
    'partition_system',
    'print_expressions',
    'register_backend',
    'resolve_backend',
]
