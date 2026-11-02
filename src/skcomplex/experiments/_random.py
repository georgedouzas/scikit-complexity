"""Define the design of independent random samples."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ._base import BaseDesign
from .types import Bounds


class RandomDesign(BaseDesign):
    """Design of independent random samples, uniform between a pair of bounds or drawn from a SciPy distribution.

    Args:
        n:
            The number of rows.

        random_state:
            The seed of the samples.

    Attributes:
        rows_:
            The `n` sampled rows, set by `prepare` on a pair `(low, high)` or a SciPy distribution per name.

    Examples:
        >>> from skcomplex.experiments import RandomDesign
        >>> len(RandomDesign(n=5, random_state=0).check().check().prepare({'r': (0.1, 1.0)}).run().rows_)
        5
    """

    def __init__(self, n: int = 64, random_state: int | None = None) -> None:
        self.n = n
        self.random_state = random_state

    def _check(self) -> None:
        """Raise an error for a number of rows that is not a nonnegative integer."""
        self._check_size(self.n)

    def _prepare(self, bounds: Bounds | None = None) -> None:
        """Set `n` rows of independent uniform samples mapped to the bounds."""
        bounds = self._check_bounds(bounds)
        uniform = np.random.default_rng(self.random_state).random((self.n, len(bounds)))
        self.rows_ = self._scale(bounds, uniform)
