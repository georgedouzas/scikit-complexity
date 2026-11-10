"""Define the types that only the engine reads."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from enum import IntEnum, StrEnum
from types import SimpleNamespace

import jax
import numpy.typing as npt
import scipy.sparse
from mesa.discrete_space import Network, OrthogonalMooreGrid, OrthogonalVonNeumannGrid
from mesa.experimental.continuous_space import ContinuousSpace as MesaContinuousSpace
from sage.all__sagemath_symbolics import Expression
from sage.structure.element import RingElement

from ..specs import ObservableSpec
from ..types import AnyArray, Array, ComplexArray, Indices, IntArray, IntTuple

type Sparse = scipy.sparse.csr_array
type Pairs = tuple[Indices, Indices]
type OptionalPairs = Pairs | None
type EdgesFound = tuple[Indices, Indices, Indices | None]
type Arrays = list[AnyArray]
type Parts = tuple[Expression, Expression]
type Chain = tuple[tuple[str, int], ...]
type Terms = list[tuple[Expression, Chain, Expression]]
type PrintedKey = tuple[int, str]
type NumericFunction = Callable[[float, Mapping[str, object]], object]
type GridFunction = Callable[..., Array]
type ObservableFunction = Callable[[float, Array, Array, Mapping[str, object]], object]
type ReadFunction = Callable[[Array, Array, OptionalPairs, float], object]
type ScalarsFunction = Callable[[float, Array, Array], list[float]]
type StepFunction = Callable[[float, Array, Array, Array], Array]
type Interpolant = Callable[[float], Array]
type BoundaryFunction = Callable[[float, Array, Array], Array]
type TracedArray = jax.Array
type TracedFunction = Callable[[object, TracedArray, TracedArray], TracedArray]
type InterpolatingStepFunction = Callable[[float, Array, float, float], tuple[Array, Interpolant]]
type AgentsFunction = Callable[[str], Sequence[object] | None]
type PairArguments = Callable[[Callable[[str], Array]], str]
type Cost = Callable[[SimpleNamespace], object] | float
type Drawn = dict[str, Array]
type CellShape = tuple[IntArray, Array, Array]
type CellGrid = tuple[CellShape, IntArray, IntArray, IntArray]
type PeriodicBox = tuple[Array, bool]
type Cone = tuple[Array, float, bool]
type CellSearch = tuple[CellGrid, Cone, PeriodicBox]
type FoundNeighbors = tuple[IntArray, Array, bool]
type Positions = Callable[..., npt.ArrayLike] | Sequence[str]
type Outline = Callable[..., Sequence[npt.ArrayLike]] | Sequence[npt.ArrayLike]
type Marks = Callable[..., npt.ArrayLike] | Sequence[Sequence[float]]
type Colors = tuple[int, int, int] | list[tuple[int, int, int, int]]
type Generators = list[tuple[str, Expression]]
type Dissipators = list[tuple[str, Expression, object]]
type Declared = dict[str, tuple[str, ObservableSpec]]
type Number = RingElement | int | float | complex
type MesaSpace = OrthogonalMooreGrid | OrthogonalVonNeumannGrid | Network | MesaContinuousSpace
type SpectralBlock = tuple[IntTuple, list[Indices], ComplexArray]


class LinearProgramBound(IntEnum):
    """A bound of a column or a row of a linear program that a compiled value gives."""

    COLUMN_LOWER = 0
    COLUMN_UPPER = 1
    ROW_LOWER = 2
    ROW_UPPER = 3


class ReadKind(StrEnum):
    """A kind of read of a symbol for each evaluation index of a kernel."""

    MEMBER = 'member'
    SINGLE = 'single'
    FIRST = 'first'
    SECOND = 'second'
    NEIGHBOR = 'neighbor'
    DERIVED = 'derived'
    REDUCED = 'reduced'
    OTHER = 'other'
    SOURCE = 'source'
