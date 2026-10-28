"""Rows of values at consecutive grid steps, kept in one array that grows or keeps only the latest rows."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ...core.types import Vector

_INITIAL_ROWS = 64


class _Rows:
    """Rows of values at consecutive grid steps, kept in one array that grows or keeps only the latest rows."""

    def __init__(self, first: Vector, keep: int | None) -> None:
        self.keep = keep
        self.values = np.zeros((_INITIAL_ROWS if keep is None else 2 * keep + 2, first.size))
        self.values[0] = first
        self.start = 0
        self.stop = 1

    def _make_room(self, size: int) -> None:
        """Double the array of a whole history, or move the kept rows of a window to its front."""
        if self.keep is None:
            grown = np.zeros((2 * size, self.values.shape[1]))
            grown[:size] = self.values
            self.values = grown
            return
        kept = self.values[size - self.keep - 1 :].copy()
        self.values[: kept.shape[0]] = kept
        self.start = self.stop - kept.shape[0]

    def append(self, row: Vector) -> None:
        """Add the row of the next step, dropping the rows outside the kept ones."""
        size = self.stop - self.start
        if size == self.values.shape[0]:
            self._make_room(size)
        self.values[self.stop - self.start] = row
        self.stop += 1

    def rows(self, first: int, last: int) -> Vector:
        """Return the rows of the steps `first` to `last - 1`."""
        return self.values[first - self.start : last - self.start]

    def row(self, step: int) -> Vector:
        """Return the row of one step."""
        return self.values[step - self.start]
