"""Define the design of Saltelli's sample, which Sobol sensitivity indices read."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from SALib.sample import sobol as sobol_sample

from ._base import BaseDesign, Bounds


class SaltelliDesign(BaseDesign):
    """Design of Saltelli's sample of SALib, the rows a Sobol sensitivity analysis reads.

    The rows are `n` base points of a Sobol sequence and, for every name, the same points with that name taken from a
    second set of points, `n * (names + 2)` rows in all.

    Args:
        n:
            The number of base points, best a power of 2.

        random_state:
            The seed of the scrambling.

    Attributes:
        rows_:
            The `n * (names + 2)` sampled rows, set by `prepare` on a pair `(low, high)` per name.

    Examples:
        >>> from skcomplex.experiments import SaltelliDesign
        >>> len(SaltelliDesign(n=8, random_state=0).check().prepare({'r': (0.0, 1.0), 'K': (50, 150)}).run())
        32
    """

    def __init__(self, n: int = 1024, random_state: int | None = None) -> None:
        self.n = n
        self.random_state = random_state

    def _check(self) -> None:
        """Raise an error for a number of base points that is not a nonnegative integer."""
        self._check_size(self.n)

    def _prepare(self, bounds: Bounds | None = None) -> None:
        """Set the rows of Saltelli's sample within the bounds."""
        bounds = self._check_bounds(bounds, distributions=False)
        names = list(bounds)
        problem = {'num_vars': len(names), 'names': names, 'bounds': [list(bounds[name]) for name in names]}  # type: ignore[call-overload]
        samples = sobol_sample.sample(problem, self.n, calc_second_order=False, seed=self.random_state)
        self.rows_ = [dict(zip(names, map(float, sample), strict=True)) for sample in samples]
