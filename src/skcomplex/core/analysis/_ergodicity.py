"""Diagnose ergodicity breaking from ensembles of runs: time averages, their fluctuations, and their distributions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..base import AnalysisError
from ._adapters import Ensemble
from ._kinds import DataKind, analysis

type Array = npt.NDArray[np.float64]

MIN_RUNS = 2
FIT_START = 0.01
FIT_POINTS = 50
BINS = 20


def _runs(values: npt.ArrayLike | Ensemble) -> Array:
    """Return one row per run and one column per time, from an ensemble or an array."""
    array = np.asarray(values.values if isinstance(values, Ensemble) else values, dtype=float)
    if array.ndim != 2 or array.shape[0] < MIN_RUNS:  # noqa: PLR2004
        error_msg = f'The diagnostics need at least two runs, one row per run, not an array of shape {array.shape}.'
        raise AnalysisError(error_msg)
    return array


def _running_averages(runs: Array) -> Array:
    """Return the time average of every run up to every time."""
    return np.cumsum(runs, axis=1) / np.arange(1, runs.shape[1] + 1)


@analysis(DataKind.ENSEMBLE)
def ergodicity_breaking(values: npt.ArrayLike | Ensemble) -> float:
    """Return the ergodicity-breaking parameter, the variance of the time averages over the variance of the values.

    The parameter tends to 0 for ergodic runs longer than their relaxation time, and stays near 1 when every run is
    frozen in its own state.

    Args:
        values:
            One row per run and one column per time, or an ensemble of an experiment.

    Returns:
        parameter:
            The variance across runs of the time averages, divided by the variance of all the values.

    Raises:
        AnalysisError:
            The values hold fewer than two runs.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import ergodicity_breaking
        >>> ergodicity_breaking([[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]])
        0.0
        >>> ergodicity_breaking([[1.0, 1.0], [-1.0, -1.0]])
        1.0
    """
    runs = _runs(values)
    spread = float(runs.var())
    return float(runs.mean(axis=1).var() / spread) if spread > 0 else 0.0


@dataclass(frozen=True)
class FluctuationMetric:
    """Thirumalai-Mountain fluctuation metric of an ensemble of runs as a function of time.

    Args:
        times:
            The times.

        metric:
            The mean square deviation of the running time averages of the runs from their ensemble mean.

        exponent:
            The slope of `log metric` against `log time` at log-spaced later times, -1 for ergodic runs and 0 for
            frozen ones.
    """

    times: Array
    metric: Array
    exponent: float


@analysis(DataKind.ENSEMBLE)
def fluctuation_metric(
    values: npt.ArrayLike | Ensemble,
    times: npt.ArrayLike | None = None,
    *,
    start: float = FIT_START,
) -> FluctuationMetric:
    """Return the fluctuation metric of Thirumalai and Mountain, which decays as `1 / t` for ergodic runs.

    Args:
        values:
            One row per run and one column per time, sampled at equal steps, or an ensemble of an experiment.

        times:
            The times of the columns, or `None` for the times of the ensemble or the steps 1, 2, and so on.

        start:
            The fraction of the run after which the exponent is fitted, at 50 log-spaced times.

    Returns:
        metric:
            The times, the metric, and its exponent.

    Raises:
        AnalysisError:
            The values hold fewer than two runs.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import fluctuation_metric
        >>> frozen = np.repeat([[1.0], [-1.0]], 100, axis=1)
        >>> found = fluctuation_metric(frozen)
        >>> float(found.metric[-1]), round(found.exponent, 6)
        (1.0, 0.0)
    """
    runs = _runs(values)
    if times is None:
        times = values.times if isinstance(values, Ensemble) else np.arange(1, runs.shape[1] + 1)
    grid = np.asarray(times, dtype=float)
    metric = _running_averages(runs).var(axis=0)
    spaced = np.unique(np.geomspace(max(int(start * grid.size), 1), grid.size, FIT_POINTS).astype(int) - 1)
    fitted = spaced[(grid[spaced] > 0) & (metric[spaced] > 0)]
    exponent = float(np.polyfit(np.log(grid[fitted]), np.log(metric[fitted]), 1)[0]) if fitted.size > 1 else 0.0
    return FluctuationMetric(grid, metric, exponent)


@analysis(DataKind.ENSEMBLE)
def ergodicity_distance(values: npt.ArrayLike | Ensemble, bins: int = BINS) -> float:
    """Return the median total-variation distance between the distribution of every run and that of the ensemble.

    The distance compares the values one run visits over time with the values of all runs, after Lesmana et al.
    (2026). It is near 0 for ergodic runs and near `1 - p` when every run stays in its own state of ensemble weight
    `p`.

    Args:
        values:
            One row per run and one column per time, such as one trait of every agent, or an ensemble of an experiment.

        bins:
            The number of bins of the histograms, shared by the runs and the ensemble.

    Returns:
        distance:
            The median over runs of half the sum of the absolute differences of the binned probabilities.

    Raises:
        AnalysisError:
            The values hold fewer than two runs.

    Examples:
        >>> from skcomplex.core.analysis import ergodicity_distance
        >>> ergodicity_distance([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
        0.0
        >>> ergodicity_distance([[0.0, 0.0], [1.0, 1.0]])
        0.5
    """
    runs = _runs(values)
    edges = np.histogram_bin_edges(runs, bins=bins)
    ensemble = np.histogram(runs, bins=edges)[0] / runs.size
    distances = [0.5 * np.abs(np.histogram(run, bins=edges)[0] / run.size - ensemble).sum() for run in runs]
    return float(np.median(distances))
