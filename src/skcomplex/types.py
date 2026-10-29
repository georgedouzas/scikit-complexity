"""Define the type aliases shared across the package."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import Enum, StrEnum
from types import SimpleNamespace
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import scipy.sparse
from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

# arrays
type Array = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type Indices = npt.NDArray[np.intp]
type ComplexArray = npt.NDArray[np.complex128]
type AnyArray = npt.NDArray[np.generic]
type Arrays = list[AnyArray]
type Sparse = scipy.sparse.csr_array
type Offset = tuple[int, ...]
type History = tuple[int, ...]
type Pair = tuple[int, int]
type Pairs = tuple[Indices, Indices]
type OptionalPairs = Pairs | None
type EdgesFound = tuple[Indices, Indices, Indices | None]
type Symmetry = tuple[Indices, tuple[int, ...], list[tuple[int, int]]]
type Partials = tuple[list[tuple[int, Expression]], Callable[..., list[Any]] | None]

# expressions and specs
type Target = Expression | FreeModuleElement
type Process = str | tuple[str, ...] | Expression | FreeModuleElement | None
type Parts = tuple[Expression, Expression]
type Chain = tuple[tuple[str, int], ...]
type Terms = list[tuple[Expression, Chain, Expression]]
type Point = Mapping[str, Expression | float]
type Printed = tuple[str, int]
type PrintedKey = tuple[int, str]

# functions of time, states, and parameters
type VectorFunction = Callable[[float, Array, Array], Array]
type NumericFunction = Callable[[float, Mapping[str, object]], object]
type GridFunction = Callable[..., Array]
type Getter = Callable[[float, Array, Array, Mapping[str, object]], object]
type Reader = Callable[[Array, Array, OptionalPairs, float], Any]
type Scalars = Callable[[float, Array, Array], list[float]]
type StepFunction = Callable[[float, Array, Array, Array], Array]
type Interpolant = Callable[[float], Array]
type Bounded = Callable[[float, Array, Array], Array]
type Take = Callable[[float, Array, float, float], tuple[Array, Interpolant]]
type Cost = Callable[[SimpleNamespace], object] | float
type Sector = tuple[Callable[[SimpleNamespace], Any], float]
type Pairing = Callable[[Sequence[object], Sequence[object], np.random.Generator, int], list[tuple[object, object]]]
type AgentsOf = Callable[[str], list[Any] | None]
type PairArguments = Callable[[Callable[[str], Array]], str]
type Drawn = dict[str, Array]
type ProcessKey = tuple[str, str, Any]

# neighbor search on a grid of cells
type Shape = tuple[IntArray, Array, Array]
type Grid = tuple[Shape, IntArray, IntArray, IntArray]
type Box = tuple[Array, bool]
type Aim = tuple[Array, float, bool]
type Context = tuple[Grid, Aim, Box]
type NeighborsFound = tuple[IntArray, Array, bool]

# analyses, experiments, and visuals
type Partition = npt.ArrayLike | Callable[[AnyArray], npt.ArrayLike]
type Row = dict[str, object]
type Bounds = Mapping[str, object]
type Positions = Callable[..., npt.ArrayLike] | Sequence[str]
type Outline = Callable[..., Sequence[npt.ArrayLike]] | Sequence[npt.ArrayLike]
type Marks = Callable[..., npt.ArrayLike] | Sequence[Sequence[float]]
type Colors = tuple[int, int, int] | list[tuple[int, int, int, int]]


class DataKind(StrEnum):
    """Kind of data an analysis consumes, whatever model or measurement produced it.

    A kind equals its value, so an analysis may return the plain string, such as `'series'`.

    Examples:
        >>> from skcomplex.types import DataKind
        >>> DataKind('field').reading
        'a grid of values, or frames of grids'
        >>> DataKind.SERIES == 'series'
        True
    """

    SERIES = 'series'
    PATH = 'path'
    FIELD = 'field'
    AGENTS = 'agents'
    NETWORK = 'network'
    ENSEMBLE = 'ensemble'
    SYSTEM = 'system'

    @property
    def reading(self) -> str:
        """Plain arrays or graphs the analyses of the kind read."""
        readings = {
            DataKind.SERIES: 'a series of values, or the columns of an array as series',
            DataKind.PATH: 'the jump times and the states after every jump',
            DataKind.FIELD: 'a grid of values, or frames of grids',
            DataKind.AGENTS: 'positions, velocities, headings, or states of agents, one row per agent',
            DataKind.NETWORK: 'a NetworkX graph, or one graph per frame',
            DataKind.ENSEMBLE: 'one row per run and one column per common time',
            DataKind.SYSTEM: (
                'a system, a problem compiled from one, or a trajectory of one, whose equations are analysed'
            ),
        }
        return readings[self]


class ReadKind(Enum):
    """How a symbol is read for each evaluation index of a kernel."""

    MEMBER = 'member'
    SINGLE = 'single'
    FIRST = 'first'
    SECOND = 'second'
    NEIGHBOR = 'neighbor'
    DERIVED = 'derived'
    REDUCED = 'reduced'
    OTHER = 'other'


class DomainKind(StrEnum):
    """Kind of value a state takes, which decides the laws that may target it."""

    REAL = 'real'
    INTEGER = 'integer'
    BOOLEAN = 'boolean'
    CATEGORICAL = 'categorical'
    ANGLE = 'angle'
    COMPLEX = 'complex'
    VECTOR = 'vector'
    TENSOR = 'tensor'
    HILBERT = 'hilbert'
    SEQUENCE = 'sequence'
    GRAPH = 'graph'
    TILING = 'tiling'


class VariableKind(Enum):
    """Role a declared argument plays in a part."""

    STATE = 'state'
    PARAMETER = 'parameter'
    ROLE = 'role'


class OrderKind(StrEnum):
    """Kind of order of the time derivative that the rate laws on a state give."""

    INTEGER = 'integer'
    FRACTIONAL = 'fractional'


class NoiseKind(StrEnum):
    """Structure of the Wiener processes that the noise laws of a system draw."""

    DIAGONAL = 'diagonal'
    SHARED = 'shared'
    CORRELATED = 'correlated'
    CONSERVED = 'conserved'


class FormKind(StrEnum):
    """Mathematical form of a compiled system, which decides the solvers that advance it.

    Examples:
        >>> from skcomplex.types import FormKind
        >>> FormKind.SDE.value
        'stochastic differential equations'
    """

    ODE = 'ordinary differential equations'
    FRACTIONAL = 'fractional differential equations'
    SDE = 'stochastic differential equations'
    JUMP = 'jump process'
    DISCRETE = 'discrete map'
    AGENT = 'agent-based model'
    LATTICE = 'lattice dynamics'
    MONTE_CARLO = 'Monte Carlo sampling'
    RENEWAL = 'renewal process'
    COUPLED = 'co-simulation'
    QUANTUM = 'quantum dynamics'


class ElementKind(StrEnum):
    """Kind of element or system that carries states, as solvers and backends support them."""

    ELEMENT = 'element'
    GROUP = 'group'
    VARIABLE_GROUP = 'variable-group'
    SITE_GROUP = 'site-group'
    EDGE_GROUP = 'edge-group'
    HYPEREDGE_GROUP = 'hyperedge-group'
    SYSTEM = 'system'


class ReservoirNodeProtocol(Protocol):
    """Node of ReservoirPy, imported on first use, as the reservoir step reads it: a reservoir or a ridge readout."""

    initialized: bool
    store: Array
    state: dict[str, Array]

    def initialize(self, x: Array) -> None:
        """Initialize the node for inputs of the shape of an example."""

    def run(self, x: Array) -> npt.ArrayLike:
        """Return the outputs of the node driven by a series of inputs."""

    def step(self, x: Array) -> npt.ArrayLike:
        """Return the output of the node after one input."""

    def fit(self, x: Array, y: Array) -> object:
        """Fit the node to inputs and targets."""
