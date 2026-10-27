"""Define designs, the rows of parameter and initial-state values that an experiment runs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import qmc

from ..core.errors import ValidationError
from ..engine.utils import BaseEstimator

type Row = dict[str, Any]


def _scale(bound: Any, uniform: np.ndarray) -> np.ndarray:  # noqa: ANN401
    """Return uniform samples in [0, 1) mapped to a pair of bounds or through a distribution's quantile function."""
    if hasattr(bound, 'ppf'):
        return np.asarray(bound.ppf(uniform), dtype=float)
    low, high = (float(item) for item in bound)
    return low + uniform * (high - low)


class BaseDesign(BaseEstimator):
    """Base class of designs, which give rows of values by factory argument or dotted path.

    Preparing a design sets its rows, and running it returns them. Designs concatenate with `concat`, which runs the
    rows of both, and cross with `product`, which runs every pair of rows.

    Attributes:
        rows_:
            One mapping from factory argument or dotted path to value per row, set by `prepare`.
    """

    rows_: list[Row]

    def _rows(self) -> list[Row]:
        """Return the rows of the design, one empty row by default."""
        return [{}]

    def _prepare(self) -> None:
        """Set the rows of the design."""
        self.rows_ = self._rows()

    def _run(self) -> list[Row]:
        """Return a copy of the prepared rows."""
        return [dict(row) for row in self.rows_]

    def concat(self, other: 'BaseDesign') -> 'RowsDesign':
        """Return the design that runs the rows of this design and then the rows of another.

        Args:
            other:
                The design whose rows follow.

        Returns:
            design:
                The design of the rows of both designs.
        """
        return RowsDesign([*self.prepare().run(), *other.prepare().run()])

    def product(self, other: 'BaseDesign') -> 'RowsDesign':
        """Return the design that runs every combination of a row of this design and a row of another.

        Args:
            other:
                The design whose rows combine with the rows of this design.

        Returns:
            design:
                The design of every combination, the values of `other` overriding shared names.
        """
        return RowsDesign([{**first, **second} for first in self.prepare().run() for second in other.prepare().run()])


class _SampledDesign(BaseDesign):
    """Design of `n` rows sampled over bounds or distributions."""

    def __init__(self, n: int, *, random_state: int | None = None, **bounds: Any) -> None:  # noqa: ANN401
        self.n = n
        self.random_state = random_state
        self.bounds = bounds

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the size, the seed, and the bounds by name."""
        del deep
        return {'n': self.n, 'random_state': self.random_state, **self.bounds}

    def _check(self) -> None:
        """Raise an error for a negative size or a bound that is neither a pair nor a distribution."""
        if isinstance(self.n, bool) or not isinstance(self.n, int | np.integer) or self.n < 0:
            error_msg = f'The number of rows of a design must be a nonnegative integer, not {self.n!r}.'
            raise ValidationError(error_msg)
        for name, bound in self.bounds.items():
            if hasattr(bound, 'ppf') or (isinstance(bound, Sequence) and len(bound) == 2):  # noqa: PLR2004
                continue
            error_msg = f'The bounds of {name} must be a pair (low, high) or a SciPy distribution, not {bound!r}.'
            raise ValidationError(error_msg)

    def _uniform(self) -> np.ndarray:
        """Return `n` uniform points in the unit cube of the bounds' dimension."""
        return np.random.default_rng(self.random_state).random((self.n, len(self.bounds)))

    def _rows(self) -> list[Row]:
        """Return one sampled row per sample."""
        uniform = self._uniform()
        columns = {name: _scale(bound, uniform[:, index]) for index, (name, bound) in enumerate(self.bounds.items())}
        return [{name: float(column[row]) for name, column in columns.items()} for row in range(self.n)]


class RowsDesign(BaseDesign):
    """Design given by its rows.

    Args:
        rows:
            One mapping from factory argument or dotted path to value per row.

    Examples:
        >>> from skcomplex.experiments import RowsDesign
        >>> RowsDesign([{'growth.r': 0.5}, {'growth.r': 1.0}]).prepare().run()
        [{'growth.r': 0.5}, {'growth.r': 1.0}]
    """

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = rows

    def _check(self) -> None:
        """Raise an error for a row that is not a mapping."""
        for row in self.rows:
            if not isinstance(row, Mapping):
                error_msg = f'Every row of a design must be a mapping from name to value, not {row!r}.'
                raise ValidationError(error_msg)

    def _rows(self) -> list[Row]:
        """Return a copy of the given rows."""
        return [dict(row) for row in self.rows]


class GridDesign(BaseDesign):
    """Design of every combination of the given values.

    Args:
        **values:
            The values of each factory argument or dotted path.

    Examples:
        >>> from skcomplex.experiments import GridDesign
        >>> len(GridDesign(r=[0.5, 1.0], K=[100, 200, 300]).prepare().run())
        6
    """

    def __init__(self, **values: Sequence[Any]) -> None:
        self.values = values

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the values by name."""
        del deep
        return dict(self.values)

    def _rows(self) -> list[Row]:
        """Return one row per combination of the values."""
        names = list(self.values)
        return [dict(zip(names, combination, strict=True)) for combination in itertools.product(*self.values.values())]


class RandomDesign(_SampledDesign):
    """Design of independent random samples, uniform between bounds or drawn from SciPy distributions.

    Args:
        n:
            The number of rows.

        random_state:
            The seed of the samples.

        **bounds:
            A pair `(low, high)` or a SciPy distribution per factory argument or dotted path.

    Examples:
        >>> from skcomplex.experiments import RandomDesign
        >>> len(RandomDesign(5, r=(0.1, 1.0), random_state=0).prepare().run())
        5
    """


class LatinHypercubeDesign(_SampledDesign):
    """Design of a Latin hypercube sample, one sample in every stratum of every dimension.

    Args:
        n:
            The number of rows.

        random_state:
            The seed of the samples.

        **bounds:
            A pair `(low, high)` or a SciPy distribution per factory argument or dotted path.

    Examples:
        >>> from skcomplex.experiments import LatinHypercubeDesign
        >>> rows = LatinHypercubeDesign(4, r=(0.0, 1.0), random_state=0).prepare().run()
        >>> sorted(int(row['r'] * 4) for row in rows)
        [0, 1, 2, 3]
    """

    def _uniform(self) -> np.ndarray:
        """Return a Latin hypercube in the unit cube."""
        return qmc.LatinHypercube(d=len(self.bounds), seed=self.random_state).random(self.n)


class SobolDesign(_SampledDesign):
    """Design of a scrambled Sobol sequence, a low-discrepancy sample, best with `n` a power of 2.

    Args:
        n:
            The number of rows.

        random_state:
            The seed of the scrambling.

        **bounds:
            A pair `(low, high)` or a SciPy distribution per factory argument or dotted path.

    Examples:
        >>> from skcomplex.experiments import SobolDesign
        >>> len(SobolDesign(8, r=(0.0, 1.0), random_state=0).prepare().run())
        8
    """

    def _uniform(self) -> np.ndarray:
        """Return a scrambled Sobol sequence in the unit cube."""
        return qmc.Sobol(d=len(self.bounds), seed=self.random_state).random(self.n)
