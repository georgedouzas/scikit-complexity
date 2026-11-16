"""Define the types that only the engine reads."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Protocol, runtime_checkable

import jax
import numpy.typing as npt
import scipy.sparse
from mesa.discrete_space import Network, OrthogonalMooreGrid, OrthogonalVonNeumannGrid
from mesa.experimental.continuous_space import ContinuousSpace as MesaContinuousSpace
from sage.all__sagemath_symbolics import Expression
from sage.structure.element import RingElement

from ..components import Observable
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
type Declared = dict[str, tuple[str, Observable]]
type Number = RingElement | int | float | complex
type MesaSpace = OrthogonalMooreGrid | OrthogonalVonNeumannGrid | Network | MesaContinuousSpace
type SpectralBlock = tuple[IntTuple, list[Indices], ComplexArray]


@runtime_checkable
class HilbertSpaceProtocol(Protocol):
    """A protocol of the state space of a subsystem that the operator compiler reads.

    A field package defines the space, and the space names the compiler of its systems.
    """

    @property
    def dimension(self) -> int:
        """The dimension of the space."""

    @property
    def operators(self) -> Mapping[str, tuple[npt.ArrayLike, bool]]:
        """The matrix of every local operator by name and whether it is Hermitian."""

    @property
    def basis(self) -> Mapping[str, int]:
        """The index of every basis state by label."""

    @property
    def display(self) -> str | None:
        """The operator whose expectation visuals show for a group, or `None`."""

    def _check(self) -> None:
        """Raise an error for a dimension, an operator, or a basis state that misfits."""

    def _get_compiler(self) -> object:
        """Return the compiler of the systems of the space."""
