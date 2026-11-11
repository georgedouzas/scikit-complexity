"""Define the types that every layer of the library reads, such as arrays."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

type Array = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type Indices = npt.NDArray[np.intp]
type ComplexArray = npt.NDArray[np.complex128]
type AnyArray = npt.NDArray[np.generic]
type IntTuple = tuple[int, ...]
type Pair = tuple[int, int]
type VectorFunction = Callable[[float, Array, Array], Array]
type ProcessKey = tuple[str, str, str | int]
type Preference = Callable[[object, object], float]
