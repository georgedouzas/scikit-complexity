"""Define the types of the engine: arrays, functions, records, and library classes that only the engine reads.

The neighbour search on a grid of cells passes tuples, since its Numba kernels take no other records.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from types import SimpleNamespace

import jax
import numpy.typing as npt
import scipy.sparse
from mesa.discrete_space import Network, OrthogonalMooreGrid, OrthogonalVonNeumannGrid
from mesa.experimental.continuous_space import ContinuousSpace as MesaContinuousSpace
from sage.all__sagemath_symbolics import Expression
from sage.structure.element import RingElement

from ..core.laws import DestructionLaw, EncounterLaw, ProliferationLaw
from ..specs import ObservableSpec
from ..types import Array, ComplexArray, Indices, IntArray, IntTuple

# arrays and indices
type Sparse = scipy.sparse.csr_array
type Pairs = tuple[Indices, Indices]
type OptionalPairs = Pairs | None
type EdgesFound = tuple[Indices, Indices, Indices | None]

# expressions
type Parts = tuple[Expression, Expression]
type Chain = tuple[tuple[str, int], ...]
type Terms = list[tuple[Expression, Chain, Expression]]
type PrintedKey = tuple[int, str]

# functions of time, states, and parameters
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

# neighbour search on a grid of cells
type CellShape = tuple[IntArray, Array, Array]
type CellGrid = tuple[CellShape, IntArray, IntArray, IntArray]
type PeriodicBox = tuple[Array, bool]
type Cone = tuple[Array, float, bool]
type CellSearch = tuple[CellGrid, Cone, PeriodicBox]
type FoundNeighbors = tuple[IntArray, Array, bool]

# visuals
type Positions = Callable[..., npt.ArrayLike] | Sequence[str]
type Outline = Callable[..., Sequence[npt.ArrayLike]] | Sequence[npt.ArrayLike]
type Marks = Callable[..., npt.ArrayLike] | Sequence[Sequence[float]]
type Colors = tuple[int, int, int] | list[tuple[int, int, int, int]]

# laws, operators, and observables the compiler reads
type KineticLaw = EncounterLaw | ProliferationLaw | DestructionLaw
type Hamiltonians = list[tuple[str, Expression]]
type Dissipators = list[tuple[str, Expression, object]]
type Declared = dict[str, tuple[str, ObservableSpec]]
type Number = RingElement | int | float | complex

# solvers backed by libraries
type MesaSpace = OrthogonalMooreGrid | OrthogonalVonNeumannGrid | Network | MesaContinuousSpace
type SpectralBlock = tuple[IntTuple, list[Indices], ComplexArray]


class ReadKind(StrEnum):
    """How a symbol is read for each evaluation index of a kernel."""

    MEMBER = 'member'
    SINGLE = 'single'
    FIRST = 'first'
    SECOND = 'second'
    NEIGHBOR = 'neighbor'
    DERIVED = 'derived'
    REDUCED = 'reduced'
    OTHER = 'other'
