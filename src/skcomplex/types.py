"""Define the type aliases shared across the package."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse
from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement

from .specs import ModuleSpec, ParameterSpec, RoleSpec, StateSpec

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
type Letters = str | ModuleSpec | Sequence[str | ModuleSpec]
type Spec = StateSpec | ParameterSpec | RoleSpec
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
