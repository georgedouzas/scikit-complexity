"""Define designs, the rows of parameter and initial-state values that an experiment runs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import qmc
from sklearn.base import BaseEstimator

from ..base import ValidationError

type Row = dict[str, Any]


class Design(BaseEstimator):
    """Base class of designs, which give rows of values by factory argument or dotted path.

    Designs concatenate with `+`, which runs the rows of both, and cross with `*`, which runs every pair of rows.
    """

    def rows(self) -> list[Row]:
        """Return the rows of the design.

        Returns:
            rows:
                One mapping from factory argument or dotted path to value per row.
        """
        return [{}]

    def __add__(self, other: 'Design') -> 'Design':
        """Return the design that runs the rows of both designs."""
        return Rows([*self.rows(), *other.rows()])

    def __mul__(self, other: 'Design') -> 'Design':
        """Return the design that runs every combination of a row of each design."""
        return Rows([{**first, **second} for first in self.rows() for second in other.rows()])


class Rows(Design):
    """Design given by its rows.

    Args:
        rows:
            One mapping from factory argument or dotted path to value per row.

    Examples:
        >>> from skcomplex.core.experiments import Rows
        >>> Rows([{'growth.r': 0.5}, {'growth.r': 1.0}]).rows()
        [{'growth.r': 0.5}, {'growth.r': 1.0}]
    """

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows_ = [dict(row) for row in rows]
        self.rows_given = rows

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the rows as the design's only parameter."""
        del deep
        return {'rows': self.rows_given}

    def rows(self) -> list[Row]:
        """Return the rows of the design.

        Returns:
            rows:
                The rows.
        """
        return [dict(row) for row in self.rows_]


class Grid(Design):
    """Design of every combination of the given values.

    Args:
        **values:
            The values of each factory argument or dotted path.

    Examples:
        >>> from skcomplex.core.experiments import Grid
        >>> len(Grid(r=[0.5, 1.0], K=[100, 200, 300]).rows())
        6
    """

    def __init__(self, **values: Sequence[Any]) -> None:
        self.values = values

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the values by name."""
        del deep
        return dict(self.values)

    def rows(self) -> list[Row]:
        """Return every combination of the values.

        Returns:
            rows:
                One row per combination.
        """
        names = list(self.values)
        return [dict(zip(names, combination, strict=True)) for combination in itertools.product(*self.values.values())]


def _scale(name: str, bound: Any, uniform: np.ndarray) -> np.ndarray:  # noqa: ANN401
    """Return uniform samples in [0, 1) mapped to a pair of bounds or through a distribution's quantile function."""
    if hasattr(bound, 'ppf'):
        return np.asarray(bound.ppf(uniform), dtype=float)
    if isinstance(bound, Sequence) and len(bound) == 2:  # noqa: PLR2004
        low, high = (float(item) for item in bound)
        return low + uniform * (high - low)
    error_msg = f'The bounds of {name} must be a pair (low, high) or a SciPy distribution, not {bound!r}.'
    raise ValidationError(error_msg)


class _Sampled(Design):
    """Design of `n` rows sampled over bounds or distributions."""

    def __init__(self, n: int, *, random_state: int | None = None, **bounds: Any) -> None:  # noqa: ANN401
        self.n = n
        self.random_state = random_state
        self.bounds = bounds

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the size, the seed, and the bounds by name."""
        del deep
        return {'n': self.n, 'random_state': self.random_state, **self.bounds}

    def _uniform(self) -> np.ndarray:
        """Return `n` uniform points in the unit cube of the bounds' dimension."""
        return np.random.default_rng(self.random_state).random((self.n, len(self.bounds)))

    def rows(self) -> list[Row]:
        """Return the sampled rows.

        Returns:
            rows:
                One row per sample.
        """
        uniform = self._uniform()
        columns = {
            name: _scale(name, bound, uniform[:, index]) for index, (name, bound) in enumerate(self.bounds.items())
        }
        return [{name: float(column[row]) for name, column in columns.items()} for row in range(self.n)]


class RandomDesign(_Sampled):
    """Design of independent random samples, uniform between bounds or drawn from SciPy distributions.

    Args:
        n:
            The number of rows.

        random_state:
            The seed of the samples.

        **bounds:
            A pair `(low, high)` or a SciPy distribution per factory argument or dotted path.

    Examples:
        >>> from skcomplex.core.experiments import RandomDesign
        >>> len(RandomDesign(5, r=(0.1, 1.0), random_state=0).rows())
        5
    """


class LatinHypercube(_Sampled):
    """Design of a Latin hypercube sample, one sample in every stratum of every dimension.

    Args:
        n:
            The number of rows.

        random_state:
            The seed of the samples.

        **bounds:
            A pair `(low, high)` or a SciPy distribution per factory argument or dotted path.

    Examples:
        >>> from skcomplex.core.experiments import LatinHypercube
        >>> rows = LatinHypercube(4, r=(0.0, 1.0), random_state=0).rows()
        >>> sorted(int(row['r'] * 4) for row in rows)
        [0, 1, 2, 3]
    """

    def _uniform(self) -> np.ndarray:
        """Return a Latin hypercube in the unit cube."""
        return qmc.LatinHypercube(d=len(self.bounds), seed=self.random_state).random(self.n)


class SobolDesign(_Sampled):
    """Design of a scrambled Sobol sequence, a low-discrepancy sample, best with `n` a power of 2.

    Args:
        n:
            The number of rows.

        random_state:
            The seed of the scrambling.

        **bounds:
            A pair `(low, high)` or a SciPy distribution per factory argument or dotted path.

    Examples:
        >>> from skcomplex.core.experiments import SobolDesign
        >>> len(SobolDesign(8, r=(0.0, 1.0), random_state=0).rows())
        8
    """

    def _uniform(self) -> np.ndarray:
        """Return a scrambled Sobol sequence in the unit cube."""
        return qmc.Sobol(d=len(self.bounds), seed=self.random_state).random(self.n)
