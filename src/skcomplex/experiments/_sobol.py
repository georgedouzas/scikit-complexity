"""Define the design of a scrambled Sobol sequence."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from scipy.stats import qmc

from ._base import BaseDesign
from .types import Bounds


class SobolDesign(BaseDesign):
    """Design of a scrambled Sobol sequence, a low-discrepancy sample, best with `n` a power of 2.

    Args:
        n:
            The number of rows.

        random_state:
            The seed of the scrambling.

    Attributes:
        rows_:
            The `n` sampled rows, set by `prepare` on a pair `(low, high)` or a SciPy distribution per name.

    Examples:
        >>> from skcomplex.experiments import SobolDesign
        >>> len(SobolDesign(n=8, random_state=0).prepare({'r': (0.0, 1.0)}).run().rows_)
        8
    """

    def __init__(self, n: int = 64, random_state: int | None = None) -> None:
        self.n = n
        self.random_state = random_state

    def _check(self) -> None:
        """Raise an error for a number of rows that is not a nonnegative integer."""
        self._check_size(self.n)

    def _prepare(self, bounds: Bounds | None = None) -> None:
        """Set `n` rows of a scrambled Sobol sequence mapped to the bounds."""
        bounds = self._check_bounds(bounds)
        self.rows_ = self._scale(bounds, qmc.Sobol(d=len(bounds), seed=self.random_state).random(self.n))
