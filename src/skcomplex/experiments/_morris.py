"""Define the design of Morris trajectories, which Morris screening reads."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from SALib.sample import morris as morris_sample

from ._base import BaseDesign, Bounds


class MorrisDesign(BaseDesign):
    """Design of the Morris trajectories of SALib on four levels, the rows a Morris sensitivity analysis reads.

    Every trajectory starts at a random point of the grid of levels and moves one name at a time, `n * (names + 1)`
    rows in all.

    Args:
        n:
            The number of trajectories.

        random_state:
            The seed of the trajectories.

    Attributes:
        rows_:
            The `n * (names + 1)` rows of the trajectories, set by `prepare` on a pair `(low, high)` per name.

    Examples:
        >>> from skcomplex.experiments import MorrisDesign
        >>> len(MorrisDesign(n=4, random_state=0).check().prepare({'r': (0.0, 1.0), 'K': (50, 150)}).run())
        12
    """

    def __init__(self, n: int = 10, random_state: int | None = None) -> None:
        self.n = n
        self.random_state = random_state

    def _check(self) -> None:
        """Raise an error for a number of trajectories that is not a nonnegative integer."""
        self._check_size(self.n)

    def _prepare(self, bounds: Bounds | None = None) -> None:
        """Set the rows of the Morris trajectories within the bounds."""
        bounds = self._check_bounds(bounds, distributions=False)
        names = list(bounds)
        problem = {'num_vars': len(names), 'names': names, 'bounds': [list(bounds[name]) for name in names]}  # type: ignore[call-overload]
        samples = morris_sample.sample(problem, self.n, num_levels=4, seed=self.random_state)
        self.rows_ = [dict(zip(names, map(float, sample), strict=True)) for sample in samples]
