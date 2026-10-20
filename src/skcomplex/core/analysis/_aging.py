"""Measure aging and response: two-time correlations, collapses on the ratio of times, and fluctuation-dissipation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..base import AnalysisError
from ._kinds import DataKind, analysis

type Array = npt.NDArray[np.float64]
GRID = 50


@analysis(DataKind.ENSEMBLE)
def two_time_correlation(
    ensemble: npt.ArrayLike,
    waiting: Sequence[int],
    lags: Sequence[int],
    *,
    connected: bool = False,
) -> Array:
    """Return the two-time correlation `C(t_w + tau, t_w) = <x(t_w) x(t_w + tau)>` of an ensemble.

    Args:
        ensemble:
            One row per run and one column per frame, with any further axes, such as sites, averaged over.

        waiting:
            The waiting times `t_w` in frames.

        lags:
            The lags `tau` in frames.

        connected:
            Whether to subtract the product of the means at the two times.

    Returns:
        correlation:
            One row per waiting time and one column per lag.

    Raises:
        AnalysisError:
            A waiting time plus a lag lies beyond the last frame.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import two_time_correlation
        >>> two_time_correlation(np.ones((4, 10)), [2], [0, 3]).tolist()
        [[1.0, 1.0]]
    """
    array = np.asarray(ensemble, dtype=float)
    array = array.reshape(array.shape[0], array.shape[1], -1)
    if max(waiting) + max(lags) >= array.shape[1]:
        error_msg = f'A waiting time plus a lag reaches frame {max(waiting) + max(lags)}, beyond {array.shape[1] - 1}.'
        raise AnalysisError(error_msg)
    offsets = np.asarray(lags, dtype=int)
    rows = []
    for start in waiting:
        first = array[:, start][:, None, :]
        later = array[:, start + offsets]
        row = np.mean(first * later, axis=(0, 2))
        rows.append(row - first.mean() * later.mean(axis=(0, 2)) if connected else row)
    return np.asarray(rows, dtype=float)


def _spread(curves: list[tuple[Array, Array]]) -> float:
    """Return the largest difference between curves interpolated on the overlap of their abscissas."""
    low = max(float(x.min()) for x, _ in curves)
    high = min(float(x.max()) for x, _ in curves)
    if low >= high:
        return float('nan')
    grid = np.linspace(low, high, GRID)
    values = np.asarray([np.interp(grid, x, y) for x, y in curves])
    return float(np.max(values.max(axis=0) - values.min(axis=0)))


@dataclass(frozen=True)
class AgingCollapse:
    """Comparison of correlation curves of several waiting times plotted against the lag and against its ratio.

    Args:
        time_spread:
            The largest difference between the curves against the lag `tau`, 0 without aging.

        ratio_spread:
            The largest difference between the curves against `tau / t_w`, 0 under simple aging.
    """

    time_spread: float
    ratio_spread: float


@analysis(DataKind.ENSEMBLE)
def aging_collapse(waiting: Sequence[float], lags: npt.ArrayLike, correlations: npt.ArrayLike) -> AgingCollapse:
    """Return how far correlation curves of several waiting times stay apart against the lag and against `tau / t_w`.

    Args:
        waiting:
            The waiting times.

        lags:
            The lags, shared by every curve.

        correlations:
            One curve per waiting time, as `two_time_correlation` returns them.

    Returns:
        collapse:
            The spreads against the lag and against the ratio.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import aging_collapse
        >>> lags = np.linspace(1, 100, 100)
        >>> curves = [np.exp(-lags / tw) for tw in (10, 100)]
        >>> collapse = aging_collapse([10, 100], lags, curves)
        >>> round(collapse.ratio_spread, 2), collapse.time_spread > 0.5
        (0.0, True)
    """
    grid = np.asarray(lags, dtype=float)
    values = np.asarray(correlations, dtype=float)
    against_time = [(grid, row) for row in values]
    against_ratio = [(grid / start, row) for start, row in zip(waiting, values, strict=True)]
    return AgingCollapse(_spread(against_time), _spread(against_ratio))


@dataclass(frozen=True)
class DistributionCollapse:
    """Distributions of several ages scaled by their age.

    Args:
        ratios:
            The common grid of `tau / t`.

        densities:
            One row per age of `t P(tau, t)` on the grid.

        spread:
            The largest relative difference between the rows where every row is positive.
    """

    ratios: Array
    densities: Array
    spread: float


@analysis(DataKind.ENSEMBLE)
def collapse_distributions(
    samples: Mapping[float, npt.ArrayLike],
    ratios: npt.ArrayLike | None = None,
) -> DistributionCollapse:
    """Return the densities `t P(tau, t)` of samples at several ages `t` against `tau / t`, which collapse under aging.

    Args:
        samples:
            The samples of `tau` at every age, such as the trapping times of the trap occupied at that age.

        ratios:
            The edges of the bins of `tau / t`, or `None` for 20 logarithmic bins from 0.01 to 10.

    Returns:
        collapse:
            The bin centres, the scaled densities, and their spread.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import collapse_distributions
        >>> rng = np.random.default_rng(0)
        >>> collapse = collapse_distributions({t: t * rng.exponential(1, 100_000) for t in (10, 100)})
        >>> bool(collapse.spread < 0.1)
        True
    """
    edges = np.geomspace(0.01, 10, 21) if ratios is None else np.asarray(ratios, dtype=float)
    ages = sorted(samples)
    densities = np.asarray(
        [
            np.histogram(np.asarray(samples[age], dtype=float) / age, bins=edges)[0]
            / (np.asarray(samples[age]).size * np.diff(edges))
            for age in ages
        ],
    )
    positive = np.all(densities > 0, axis=0)
    spread = float('nan')
    if positive.any():
        kept = densities[:, positive]
        spread = float(np.max((kept.max(axis=0) - kept.min(axis=0)) / kept.mean(axis=0)))
    return DistributionCollapse(np.sqrt(edges[1:] * edges[:-1]), densities, spread)


@analysis(DataKind.ENSEMBLE)
def response(perturbed: npt.ArrayLike, unperturbed: npt.ArrayLike, field: float) -> Array:
    """Return the integrated response `chi(t) = (<x(t)>_h - <x(t)>_0) / h` of ensembles with and without a field.

    Args:
        perturbed:
            One row per run and one column per frame, with the field switched on.

        unperturbed:
            The same without the field.

        field:
            The strength of the field.

    Returns:
        response:
            The integrated response at every frame.

    Examples:
        >>> from skcomplex.core.analysis import response
        >>> response([[1.0, 2.0]], [[1.0, 1.0]], 0.5).tolist()
        [0.0, 2.0]
    """
    first, second = np.asarray(perturbed, dtype=float), np.asarray(unperturbed, dtype=float)
    return (first.mean(axis=0) - second.mean(axis=0)) / field


@analysis(DataKind.ENSEMBLE)
def fluctuation_dissipation_ratio(
    correlation: npt.ArrayLike,
    susceptibility: npt.ArrayLike,
    temperature: float,
) -> float:
    """Return the fluctuation-dissipation ratio, the slope of `T chi` against `C(t_w, t_w) - C(t, t_w)`.

    Args:
        correlation:
            The two-time correlation over lags from 0.

        susceptibility:
            The integrated response over the same lags.

        temperature:
            The temperature.

    Returns:
        ratio:
            1 in equilibrium, below 1 in an aging system.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import fluctuation_dissipation_ratio
        >>> lags = np.linspace(0, 3, 30)
        >>> round(fluctuation_dissipation_ratio(np.exp(-lags), 1 - np.exp(-lags), 1.0), 6)
        1.0
    """
    values = np.asarray(correlation, dtype=float)
    decay = values[0] - values
    scaled = temperature * np.asarray(susceptibility, dtype=float)
    return float(np.sum(decay * scaled) / np.sum(decay**2))
