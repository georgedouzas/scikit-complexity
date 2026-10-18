"""Compile systems into problems of one mathematical form with fast numeric functions."""

from ._backends import BACKENDS, Backend, Lowering, NumpyBackend, register_backend, resolve_backend
from ._compiler import compile_system, partition_system
from ._layout import Entry, Layout
from ._numba_backend import NumbaBackend
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
    'Accelerated',
    'Backend',
    'CompiledAgents',
    'CompiledElement',
    'CompiledInteraction',
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
    'partition_system',
    'register_backend',
    'resolve_backend',
]
