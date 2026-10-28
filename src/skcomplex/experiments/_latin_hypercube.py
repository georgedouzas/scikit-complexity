"""Define the design of a Latin hypercube sample."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from scipy.stats import qmc

from ._base import BaseDesign, Bounds


class LatinHypercubeDesign(BaseDesign):
    """Design of a Latin hypercube sample, one sample in every stratum of every name.

    Args:
        n:
            The number of rows.

        random_state:
            The seed of the samples.

    Attributes:
        rows_:
            The `n` sampled rows, set by `prepare` on a pair `(low, high)` or a SciPy distribution per name.

    Examples:
        >>> from skcomplex.experiments import LatinHypercubeDesign
        >>> rows = LatinHypercubeDesign(n=4, random_state=0).check().prepare({'r': (0.0, 1.0)}).run()
        >>> sorted(int(row['r'] * 4) for row in rows)
        [0, 1, 2, 3]
    """

    def __init__(self, n: int = 64, random_state: int | None = None) -> None:
        self.n = n
        self.random_state = random_state

    def _check(self) -> None:
        """Raise an error for a number of rows that is not a nonnegative integer."""
        self._check_size(self.n)

    def _prepare(self, bounds: Bounds | None = None) -> None:
        """Set `n` rows of a Latin hypercube mapped to the bounds."""
        bounds = self._check_bounds(bounds)
        self.rows_ = self._scale(bounds, qmc.LatinHypercube(d=len(bounds), seed=self.random_state).random(self.n))
