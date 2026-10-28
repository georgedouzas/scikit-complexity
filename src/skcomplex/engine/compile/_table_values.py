"""Values of selected entries of a table of numbers or parameter expressions at a parameter vector."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ..problems import numpy_function

type Vector = npt.NDArray[np.float64]


type Indices = npt.NDArray[np.intp]


@dataclass
class TableValues:
    """Values of selected entries of a table of numbers or parameter expressions at a parameter vector.

    Args:
        table:
            The table, a float array or an object array of Sage expressions.

        positions:
            The indices of the selected entries, one array per dimension.

        parameters:
            The flat index in the parameter vector of every parameter symbol the selected entries read.

        time:
            The time symbol.
    """

    table: npt.NDArray[Any]
    positions: tuple[Indices, ...]
    parameters: Mapping[Expression, int]
    time: Expression

    @functools.cached_property
    def _selected(self) -> npt.NDArray[Any]:
        """Selected entries of the table."""
        return self.table[self.positions] if self.table.ndim else np.full(self.positions[0].size, self.table[()])

    @functools.cached_property
    def _constant(self) -> Vector | None:
        """Selected entries of a table of numbers, or `None` for a table of expressions."""
        return np.asarray(self._selected, dtype=float) if self._selected.dtype != object else None

    @functools.cached_property
    def _symbols(self) -> list[Expression]:
        """Parameter symbols the selected entries read, sorted by name."""
        return sorted(self.parameters, key=str)

    @functools.cached_property
    def _indices(self) -> Indices:
        """Flat index in the parameter vector of every symbol the selected entries read."""
        return np.asarray([self.parameters[item] for item in self._symbols], dtype=np.intp)

    @functools.cached_property
    def _function(self) -> Callable[..., list[Any]]:
        """NumPy function of the selected entries of a table of expressions, compiled once."""
        return numpy_function(self.time, self._symbols, list(self._selected))

    def values(self, p: Vector) -> Vector:
        """Return the values of the selected entries.

        Args:
            p:
                The parameter vector.

        Returns:
            values:
                One value per selected entry.
        """
        if self._constant is not None:
            return self._constant
        return np.asarray(self._function(0.0, *(float(value) for value in p[self._indices])), dtype=float)
