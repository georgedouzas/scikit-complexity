"""Measure invariants of dynamics: correlation dimensions, Lyapunov-derived invariants, and invariant densities."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

import nolds
import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from ..base import AnalysisError
from ._kinds import DataKind, analysis
from ._transitions import stationary_distribution

type Array = npt.NDArray[np.float64]
RADII = 30
PAIR_SAMPLE = 200_000


@dataclass(frozen=True)
class CorrelationDimension:
    """Correlation sums of a point set over radii and the dimension fitted in their scaling region.

    Args:
        radii:
            The radii.

        sums:
            The share of pairs of points closer than every radius.

        dimension:
            The slope of `log C(r)` against `log r` in the scaling region.

        fit_range:
            The smallest and largest radii of the fit.
    """

    radii: Array
    sums: Array
    dimension: float
    fit_range: tuple[float, float]


def _pair_distances(points: Array, rng: np.random.Generator) -> Array:
    """Return the distances of a random sample of pairs of distinct points."""
    first = rng.integers(0, points.shape[0], PAIR_SAMPLE)
    second = rng.integers(0, points.shape[0], PAIR_SAMPLE)
    distinct = first != second
    return np.linalg.norm(points[first[distinct]] - points[second[distinct]], axis=1)


@analysis(DataKind.SERIES)
def correlation_dimension(
    points: npt.ArrayLike,
    radii: npt.ArrayLike | None = None,
    *,
    fit_sums: tuple[float, float] = (1e-4, 1e-2),
    random_state: int | None = 0,
) -> CorrelationDimension:
    """Return the Grassberger-Procaccia correlation dimension of a point set, from tree counts of close pairs.

    Args:
        points:
            One row per point, such as an embedded series or agent positions.

        radii:
            The radii, or `None` for 30 radii spaced logarithmically over the range of pair distances.

        fit_sums:
            The smallest and largest correlation sums of the scaling region the slope is fitted in.

        random_state:
            The seed of the sample of pairs that sets the default radii.

    Returns:
        dimension:
            The radii, the correlation sums, the dimension, and the fit range.

    Raises:
        AnalysisError:
            Fewer than 2 radii lie in the scaling region.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import correlation_dimension
        >>> points = np.random.default_rng(0).random((5000, 2))
        >>> round(correlation_dimension(points).dimension, 1)
        2.0
    """
    array = np.asarray(points, dtype=float)
    array = array.reshape(array.shape[0], -1)
    size = array.shape[0]
    if radii is None:
        distances = _pair_distances(array, np.random.default_rng(random_state))
        low, high = np.quantile(distances, [fit_sums[0] / 10, 0.5])
        radii = np.geomspace(max(low, 1e-12), high, RADII)
    grid = np.asarray(radii, dtype=float)
    tree = cKDTree(array)
    sums = (tree.count_neighbors(tree, grid) - size) / (size * (size - 1))
    inside = np.flatnonzero((sums >= fit_sums[0]) & (sums <= fit_sums[1]))
    if inside.size < 2:  # noqa: PLR2004
        error_msg = f'Fewer than 2 radii have correlation sums within {fit_sums}. Give more radii or points.'
        raise AnalysisError(error_msg)
    slope = float(np.polyfit(np.log(grid[inside]), np.log(sums[inside]), 1)[0])
    return CorrelationDimension(grid, sums, slope, (float(grid[inside[0]]), float(grid[inside[-1]])))


@analysis(DataKind.SERIES)
def kaplan_yorke_dimension(exponents: npt.ArrayLike) -> float:
    """Return the Kaplan-Yorke dimension `j + Σ_{i<=j} λ_i / |λ_{j+1}|` of a Lyapunov spectrum.

    Args:
        exponents:
            The Lyapunov exponents.

    Returns:
        dimension:
            The dimension, with `j` the largest number of exponents whose sum is nonnegative.

    Examples:
        >>> from skcomplex.core.analysis import kaplan_yorke_dimension
        >>> round(kaplan_yorke_dimension([0.906, 0.0, -14.572]), 2)
        2.06
    """
    ordered = np.sort(np.asarray(exponents, dtype=float))[::-1]
    sums = np.cumsum(ordered)
    nonnegative = np.flatnonzero(sums >= 0)
    if not nonnegative.size:
        return 0.0
    count = int(nonnegative[-1]) + 1
    if count == ordered.size:
        return float(count)
    return float(count + sums[count - 1] / abs(ordered[count]))


@analysis(DataKind.SERIES)
def kolmogorov_sinai_entropy(exponents: npt.ArrayLike) -> float:
    """Return the Kolmogorov-Sinai entropy by Pesin's identity, the sum of the positive Lyapunov exponents.

    Args:
        exponents:
            The Lyapunov exponents.

    Returns:
        entropy:
            The entropy rate of the dynamics in nats per unit time.

    Examples:
        >>> from skcomplex.core.analysis import kolmogorov_sinai_entropy
        >>> kolmogorov_sinai_entropy([0.9, 0.0, -14.5])
        0.9
    """
    array = np.asarray(exponents, dtype=float)
    return float(array[array > 0].sum())


@analysis(DataKind.SERIES)
def largest_lyapunov_exponent(
    values: npt.ArrayLike,
    dt: float = 1.0,
    *,
    dimension: int = 3,
    delay: int = 1,
    min_separation: int | None = None,
    trajectory_length: int = 20,
    skip: int = 0,
) -> float:
    """Return Rosenstein's estimate of the largest Lyapunov exponent from one recorded series.

    Args:
        values:
            The series.

        dt:
            The time between samples.

        dimension:
            The dimension of the delay embedding.

        delay:
            The delay of the embedding in steps.

        min_separation:
            The fewest steps between neighbours, or `None` for the mean period from the spectrum.

        trajectory_length:
            The number of steps over which the divergence of neighbours is followed and fitted.

        skip:
            The number of first steps of the divergence left out of the fit, while neighbours align with the unstable
            direction.

    Returns:
        exponent:
            The largest exponent per unit time.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import largest_lyapunov_exponent
        >>> x = np.empty(5000)
        >>> x[0] = 0.3
        >>> for t in range(1, x.size):
        ...     x[t] = 4 * x[t - 1] * (1 - x[t - 1])
        >>> round(largest_lyapunov_exponent(x, dimension=2, trajectory_length=4, min_separation=10), 1)
        0.7
    """
    array = np.asarray(values, dtype=float)
    exponent = nolds.lyap_r(
        array,
        emb_dim=dimension,
        lag=delay,
        min_tsep=min_separation,
        tau=dt,
        trajectory_len=trajectory_length,
        fit='poly',
        fit_offset=skip,
    )
    return float(exponent)


@analysis(DataKind.SERIES)
def invariant_density(
    values: npt.ArrayLike,
    bins: int | npt.ArrayLike = 100,
    value_range: tuple[float, float] | None = None,
) -> tuple[Array, Array]:
    """Return the invariant density of a map or a flow estimated by the histogram of a long run.

    Args:
        values:
            The values of one coordinate along the run.

        bins:
            The number of bins or their edges.

        value_range:
            The range of the bins, or `None` for the range of the values.

    Returns:
        density:
            The edges of the bins and the density in every bin.

    Examples:
        >>> from skcomplex.core.analysis import invariant_density
        >>> invariant_density([0.1, 0.2, 0.6, 0.7], 2, (0.0, 1.0))[1].tolist()
        [1.0, 1.0]
    """
    density, edges = np.histogram(np.asarray(values, dtype=float), bins=bins, range=value_range, density=True)
    return edges, density


@dataclass(frozen=True)
class UlamChain:
    """Markov chain of a map coarse-grained on a partition of an interval by Ulam's method.

    Args:
        edges:
            The edges of the cells.

        matrix:
            The probability that a point of every cell lands in every cell.

        density:
            The stationary vector of the chain divided by the cell widths, which approximates the invariant density.
    """

    edges: Array
    matrix: Array
    density: Array


@analysis(DataKind.SERIES)
def ulam_chain(
    mapping: Callable[[Array], Array],
    edges: npt.ArrayLike,
    samples: int = 100,
) -> UlamChain:
    """Return the Ulam chain of a one-dimensional map, the share of every cell that the map sends to every cell.

    Args:
        mapping:
            The map, applied to an array of points.

        edges:
            The increasing edges of the cells.

        samples:
            The number of evenly spaced points per cell.

    Returns:
        chain:
            The edges, the transition matrix, and the density of its stationary vector.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import ulam_chain
        >>> chain = ulam_chain(lambda x: np.mod(2 * x, 1), np.linspace(0, 1, 5))
        >>> chain.density.round(2).tolist()
        [1.0, 1.0, 1.0, 1.0]
    """
    grid = np.asarray(edges, dtype=float)
    cells = grid.size - 1
    offsets = (np.arange(samples) + 0.5) / samples
    points = grid[:-1, None] + np.diff(grid)[:, None] * offsets[None, :]
    landed = np.clip(np.searchsorted(grid, np.asarray(mapping(points.ravel())), side='right') - 1, 0, cells - 1)
    origins = np.repeat(np.arange(cells), samples)
    matrix = np.bincount(origins * cells + landed, minlength=cells * cells).reshape(cells, cells) / samples
    return UlamChain(grid, matrix, stationary_distribution(matrix) / np.diff(grid))
