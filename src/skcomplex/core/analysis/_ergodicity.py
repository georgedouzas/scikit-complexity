"""Diagnose ergodicity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..errors import AnalysisError
from ..specs import AnalysisTags
from ..types import Array, DataKind
from ._base import _check_choice, _check_settings, _read_series
from ._markov import _stationary_distribution
from .types import Source

MIN_RUNS = 2
FIT_POINTS = 50
KINDS = ('breaking', 'metric', 'distance', 'invariant-density', 'ulam')
ENSEMBLE_KINDS = ('breaking', 'metric', 'distance')
SETTINGS = {
    'breaking': set(),
    'metric': {'start', 'dt'},
    'distance': {'bins'},
    'invariant-density': {'bins', 'value_range'},
    'ulam': {'edges', 'samples'},
}
BINS = {'distance': 20, 'invariant-density': 100}
ENSEMBLE_DIMENSIONS = 2
TAGS = AnalysisTags(DataKind.SERIES)
KIND_TAGS = {
    'breaking': AnalysisTags(DataKind.ENSEMBLE),
    'metric': AnalysisTags(DataKind.ENSEMBLE),
    'distance': AnalysisTags(DataKind.ENSEMBLE),
}


@dataclass(frozen=True)
class ErgodicityResult:
    """A result of `ergodicity`, a diagnostic of ergodicity.

    The diagnostic is of one kind.

    Args:
        values:
            The runs, one row per run, or the values of the run.

        mapping:
            The map of `'ulam'`.

        ergodicity_breaking:
            The ergodicity-breaking parameter, of `'breaking'`.

        times:
            The times of the columns, of `'metric'`.

        metric:
            The fluctuation metric at every time, of `'metric'`.

        exponent:
            The slope of `log metric` against `log time` at log-spaced later times, -1 for ergodic runs and 0 for
            frozen ones, of `'metric'`.

        ergodicity_distance:
            The median distance between the law of every run and that of the ensemble, of `'distance'`.

        edges:
            The edges of the bins or of the cells, of the density kinds.

        density:
            The density in every bin or cell, of the density kinds.

        transition_matrix:
            The probability that a point of every cell lands in every cell, of `'ulam'`.
    """

    values: Array | None = None
    mapping: Callable[[Array], Array] | None = None
    ergodicity_breaking: float | None = None
    times: Array | None = None
    metric: Array | None = None
    exponent: float | None = None
    ergodicity_distance: float | None = None
    edges: Array | None = None
    density: Array | None = None
    transition_matrix: Array | None = None


def ergodicity(
    source: Source | Callable[[Array], Array],
    *,
    kind: str = 'breaking',
    variables: Sequence[str] | None = None,
    start: float = 0.01,
    dt: float = 1.0,
    bins: int | npt.ArrayLike | None = None,
    value_range: tuple[float, float] | None = None,
    edges: npt.ArrayLike | None = None,
    samples: int = 100,
) -> ErgodicityResult:
    """Return a diagnostic of the ergodicity of an ensemble of runs, one run, or a map.

    The diagnostic is of one kind.

    The ensemble kinds read one row per run and one column per time, such as `ExperimentResult.get_ensemble` returns:

    - `'breaking'`: the variance across runs of the time averages over the variance of all the values, near 0 for
      ergodic runs longer than their relaxation time and near 1 when every run is frozen in its own state.
    - `'metric'`: the fluctuation metric of Thirumalai and Mountain, the mean square deviation of the running time
      averages from their ensemble mean, which decays as `1 / t` for ergodic runs and levels off for frozen ones.
    - `'distance'`: the median total-variation distance between the values one run visits over time and the values
      of all runs, after Lesmana et al. (2026), near 0 for ergodic runs and near `1 - p` when every run stays in its
      own state of ensemble weight `p`.

    The density kinds read one coordinate of a long run, or a one-dimensional map:

    - `'invariant-density'`: the histogram density of the values of a long run.
    - `'ulam'`: the Markov chain of a map coarse-grained on the cells between `edges` by Ulam's method, and the
      density of its stationary vector.

    It appears in many fields. It asks whether a time average equals an ensemble average, as for the growth of wealth in
    economics and the motion of a gas in physics.

    Args:
        source:
            The runs of an ensemble, one row per run, for the ensemble kinds, the trajectory or the values of one run
            for `'invariant-density'`, or the map, a function of an array of points, for `'ulam'`.

        kind:
            The diagnostic, one of the kinds above.

        variables:
            The variable of a trajectory whose values `'invariant-density'` reads, or `None` for the first one.

        start:
            The fraction of the run after which the exponent of the metric is fitted, at 50 log-spaced times.

        dt:
            The time between the columns of the runs.

        bins:
            The number of bins or their edges, or `None` for 20 for `'distance'` and 100 for `'invariant-density'`.

        value_range:
            The range of the bins, or `None` for the range of the values.

        edges:
            The increasing edges of the cells of `'ulam'`.

        samples:
            The number of evenly spaced points per cell of `'ulam'`.

    Returns:
        result:
            The values read and the diagnostic of the kind.

    Raises:
        AnalysisError:
            The kind is unknown, a setting the kind does not read is changed, a setting is out of range, the edges of
            the Ulam chain are missing, the map is not callable, or the ensemble has fewer than two runs.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import ergodicity
        >>> runs = [[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]]
        >>> ergodicity(runs).ergodicity_breaking
        0.0
        >>> ergodicity([[0.0, 0.0], [1.0, 1.0]], kind='distance').ergodicity_distance
        0.5
        >>> ulam = ergodicity(lambda x: np.mod(2 * x, 1), kind='ulam', edges=np.linspace(0, 1, 5))
        >>> ulam.density.round(2).tolist()
        [1.0, 1.0, 1.0, 1.0]
    """
    _check_choice('ergodicity', 'kind', kind, KINDS)
    _check_settings(
        ergodicity,
        'kind',
        SETTINGS[kind] | {'variables'},
        kind=kind,
        variables=variables,
        start=start,
        dt=dt,
        bins=bins,
        value_range=value_range,
        edges=edges,
        samples=samples,
    )
    if kind == 'ulam' and edges is None:
        error_msg = f'ergodicity of kind {kind!r} needs the edges of the cells. Give edges.'
        raise AnalysisError(error_msg)
    if samples < 1 or dt <= 0:
        error_msg = f'ergodicity has samples={samples!r} and dt={dt!r}. Use positive values.'
        raise AnalysisError(error_msg)
    if kind == 'ulam':
        if not callable(source):
            error_msg = (
                f'ergodicity of kind ulam reads a map, not {type(source).__name__}. Pass a function of an array of '
                'points.'
            )
            raise AnalysisError(error_msg)
        return _ulam(source, np.asarray(edges, dtype=float), samples)
    if callable(source):
        error_msg = f'ergodicity of kind {kind!r} reads data, not a function.'
        raise AnalysisError(error_msg)
    if kind == 'invariant-density':
        values = _read_series(source, variables).values[:, 0].astype(float)
        count = BINS['invariant-density'] if bins is None else bins
        density, found = np.histogram(values, bins=count, range=value_range, density=True)
        return ErgodicityResult(values=values, edges=found, density=density)
    runs = np.asarray(source, dtype=float)
    times = np.arange(1, runs.shape[-1] + 1) * dt
    if runs.ndim != ENSEMBLE_DIMENSIONS or runs.shape[0] < MIN_RUNS:
        error_msg = f'The diagnostics need at least two runs, one row per run, not an array of shape {runs.shape}.'
        raise AnalysisError(error_msg)
    if kind == 'breaking':
        spread = float(runs.var())
        breaking = float(runs.mean(axis=1).var() / spread) if spread > 0 else 0.0
        return ErgodicityResult(values=runs, times=times, ergodicity_breaking=breaking)
    if kind == 'metric':
        metric = (np.cumsum(runs, axis=1) / np.arange(1, runs.shape[1] + 1)).var(axis=0)
        spaced = np.unique(np.geomspace(max(int(start * times.size), 1), times.size, FIT_POINTS).astype(int) - 1)
        fitted = spaced[(times[spaced] > 0) & (metric[spaced] > 0)]
        exponent = float(np.polyfit(np.log(times[fitted]), np.log(metric[fitted]), 1)[0]) if fitted.size > 1 else 0.0
        return ErgodicityResult(values=runs, times=times, metric=metric, exponent=exponent)
    found = np.histogram_bin_edges(runs, bins=BINS['distance'] if bins is None else bins)
    ensemble = np.histogram(runs, bins=found)[0] / runs.size
    distances = [0.5 * np.abs(np.histogram(run, bins=found)[0] / run.size - ensemble).sum() for run in runs]
    return ErgodicityResult(values=runs, times=times, ergodicity_distance=float(np.median(distances)))


def _ulam(mapping: Callable[[Array], Array], grid: Array, samples: int) -> ErgodicityResult:
    """Return the Ulam chain of the map and the density of its stationary vector."""
    cells = grid.size - 1
    offsets = (np.arange(samples) + 0.5) / samples
    points = grid[:-1, None] + np.diff(grid)[:, None] * offsets[None, :]
    landed = np.searchsorted(grid, np.asarray(mapping(points.ravel())), side='right') - 1
    landed = np.clip(landed, 0, cells - 1)
    origins = np.repeat(np.arange(cells), samples)
    counts = np.bincount(origins * cells + landed, minlength=cells * cells).reshape(cells, cells)
    transition_matrix = counts / samples
    return ErgodicityResult(
        mapping=mapping,
        edges=grid,
        density=_stationary_distribution(transition_matrix) / np.diff(grid),
        transition_matrix=transition_matrix,
    )
