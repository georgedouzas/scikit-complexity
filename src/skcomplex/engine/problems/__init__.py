"""Define what a system compiles to: the problem, and the kernels its numeric functions are written from."""

from ._domains import contains, decode, domain_kind, encode, labels_of, wrap_angle
from ._hybrid import CompiledEvent, EventRecord
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
from ._kinds import DomainKind, ElementKind, Form, NoiseKind, OrderKind
from ._layouts import Entry, Layout
from ._printing import NUMPY_DIALECT, SCALAR_DIALECT, Dialect, numpy_function, print_expressions
from ._problem import (
    Accelerated,
    BaseProblem,
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
)
from ._quantities import LoopTerm, StockFlow
from ._reductions import ReductionSchedule
from ._steps import step_count
from ._support import SupportSpec
from ._symbols import components_of, domain_of, symbol, symbols_of, vector
from ._variables import Variable, VariableKind, declarations_of, declared_variables, default_of

__all__: list[str] = [
    'LATTICE_BOUNDARIES',
    'MODES',
    'NUMPY_DIALECT',
    'SCALAR_DIALECT',
    'Accelerated',
    'BaseProblem',
    'CompiledAgents',
    'CompiledElement',
    'CompiledEvent',
    'CompiledInteraction',
    'Dialect',
    'DomainKind',
    'Draws',
    'EdgeSearch',
    'ElementKind',
    'Entry',
    'EventRecord',
    'Form',
    'FractionalRows',
    'Getter',
    'Kernel',
    'Layout',
    'LoopTerm',
    'Lowered',
    'NeighborRead',
    'NetworkSnapshot',
    'NoiseKind',
    'NoiseProcess',
    'OrderKind',
    'Output',
    'Partition',
    'Problem',
    'Record',
    'ReductionSchedule',
    'Reference',
    'RenewalRecord',
    'Source',
    'StockFlow',
    'SupportSpec',
    'Variable',
    'VariableKind',
    'VectorFunction',
    'components_of',
    'contains',
    'declarations_of',
    'declared_variables',
    'decode',
    'default_of',
    'domain_kind',
    'domain_of',
    'encode',
    'labels_of',
    'numpy_function',
    'print_expressions',
    'step_count',
    'symbol',
    'symbols_of',
    'vector',
    'wrap_angle',
]
