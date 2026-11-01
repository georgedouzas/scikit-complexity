"""Diagnose ergodicity: time averages of ensembles of runs, and invariant densities of maps and flows."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from ..errors import AnalysisError
from ..types import Array, DataKind
from ._base import BaseAnalysis
from ._markov import MarkovAnalysis
from .types import Source

ENSEMBLE_DIMENSIONS = 2

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
"""Settings every kind reads, besides `kind` and `variables`."""

BINS = {'distance': 20, 'invariant-density': 100}
"""The number of bins of every kind without `bins`."""


class ErgodicityAnalysis(BaseAnalysis):
    """Ergodicity of an ensemble of runs, or invariant density of a map or a flow, of one kind.

    The ensemble kinds read one row per run and one column per time, such as `ExperimentResult.ensemble` returns:

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

    Args:
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

    Attributes:
        values_:
            The runs, one row per run, or the values of the run.

        mapping_:
            The map of `'ulam'`.

        ergodicity_breaking_:
            The ergodicity-breaking parameter, of `'breaking'`.

        times_:
            The times of the columns, of `'metric'`.

        metric_:
            The fluctuation metric at every time, of `'metric'`.

        exponent_:
            The slope of `log metric` against `log time` at log-spaced later times, -1 for ergodic runs and 0 for
            frozen ones, of `'metric'`.

        ergodicity_distance_:
            The median distance between the law of every run and that of the ensemble, of `'distance'`.

        edges_:
            The edges of the bins or of the cells, of the density kinds.

        density_:
            The density in every bin or cell, of the density kinds.

        transition_matrix_:
            The probability that a point of every cell lands in every cell, of `'ulam'`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import ErgodicityAnalysis
        >>> runs = [[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]]
        >>> ErgodicityAnalysis().check().prepare(runs).run().ergodicity_breaking_
        0.0
        >>> ErgodicityAnalysis('distance').check().prepare([[0.0, 0.0], [1.0, 1.0]]).run().ergodicity_distance_
        0.5
        >>> ulam = ErgodicityAnalysis('ulam', edges=np.linspace(0, 1, 5))
        >>> ulam.check().prepare(lambda x: np.mod(2 * x, 1)).run().density_.round(2).tolist()
        [1.0, 1.0, 1.0, 1.0]
    """

    def __init__(
        self,
        kind: str = 'breaking',
        variables: Sequence[str] | None = None,
        *,
        start: float = 0.01,
        dt: float = 1.0,
        bins: int | npt.ArrayLike | None = None,
        value_range: tuple[float, float] | None = None,
        edges: npt.ArrayLike | None = None,
        samples: int = 100,
    ) -> None:
        self.kind = kind
        self.variables = variables
        self.start = start
        self.dt = dt
        self.bins = bins
        self.value_range = value_range
        self.edges = edges
        self.samples = samples

    def _check(self) -> None:
        """Validate the kind, the edges of the Ulam chain, and that the kind reads every changed setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_settings('kind', SETTINGS[self.kind] | {'variables'})
        if self.kind == 'ulam' and self.edges is None:
            error_msg = f'{type(self).__name__} of kind {self.kind!r} needs the edges of the cells. Give edges.'
            raise AnalysisError(error_msg)
        if self.samples < 1 or self.dt <= 0:
            error_msg = f'{type(self).__name__} has samples={self.samples!r} and dt={self.dt!r}. Use positive values.'
            raise AnalysisError(error_msg)

    def _get_data_kind(self) -> DataKind:
        """Return the ensemble kind for runs, and the series kind for densities."""
        return DataKind.ENSEMBLE if self.kind in ENSEMBLE_KINDS else DataKind.SERIES

    def _prepare(self, source: Source | Callable[[Array], Array]) -> None:
        """Read the runs of an ensemble, one row per run, the values of a run, or the map."""
        if self.kind == 'ulam':
            if not callable(source):
                error_msg = (
                    f'{type(self).__name__} of kind ulam reads a map, not {type(source).__name__}. Prepare it on a '
                    'function of an array of points.'
                )
                raise AnalysisError(error_msg)
            self.mapping_ = source
            return
        if self.kind == 'invariant-density':
            self.values_ = self._read_series(source, self.variables).values[:, 0].astype(float)
            return
        self.values_ = np.asarray(source, dtype=float)
        self.times_ = np.arange(1, self.values_.shape[-1] + 1) * self.dt
        if self.values_.ndim != ENSEMBLE_DIMENSIONS or self.values_.shape[0] < MIN_RUNS:
            error_msg = (
                f'The diagnostics need at least two runs, one row per run, not an array of shape {self.values_.shape}.'
            )
            raise AnalysisError(error_msg)

    def _run_metric(self) -> None:
        """Compute the fluctuation metric and its exponent."""
        runs, grid = self.values_, self.times_
        self.metric_ = (np.cumsum(runs, axis=1) / np.arange(1, runs.shape[1] + 1)).var(axis=0)
        spaced = np.unique(np.geomspace(max(int(self.start * grid.size), 1), grid.size, FIT_POINTS).astype(int) - 1)
        fitted = spaced[(grid[spaced] > 0) & (self.metric_[spaced] > 0)]
        self.exponent_ = (
            float(np.polyfit(np.log(grid[fitted]), np.log(self.metric_[fitted]), 1)[0]) if fitted.size > 1 else 0.0
        )

    def _run_distance(self) -> None:
        """Compute the median distance between the law of every run and that of the ensemble."""
        runs = self.values_
        edges = np.histogram_bin_edges(runs, bins=BINS['distance'] if self.bins is None else self.bins)
        ensemble = np.histogram(runs, bins=edges)[0] / runs.size
        distances = [0.5 * np.abs(np.histogram(run, bins=edges)[0] / run.size - ensemble).sum() for run in runs]
        self.ergodicity_distance_ = float(np.median(distances))

    def _run_ulam(self) -> None:
        """Compute the Ulam chain of the map and the density of its stationary vector."""
        grid = np.asarray(self.edges, dtype=float)
        cells = grid.size - 1
        offsets = (np.arange(self.samples) + 0.5) / self.samples
        points = grid[:-1, None] + np.diff(grid)[:, None] * offsets[None, :]
        landed = np.searchsorted(grid, np.asarray(self.mapping_(points.ravel())), side='right') - 1
        landed = np.clip(landed, 0, cells - 1)
        origins = np.repeat(np.arange(cells), self.samples)
        counts = np.bincount(origins * cells + landed, minlength=cells * cells).reshape(cells, cells)
        self.edges_ = grid
        self.transition_matrix_ = counts / self.samples
        self.density_ = MarkovAnalysis._stationary_distribution(self.transition_matrix_) / np.diff(grid)

    def _run(self, inputs: object = None) -> None:
        """Compute the diagnostic of the kind."""
        if self.kind == 'breaking':
            spread = float(self.values_.var())
            self.ergodicity_breaking_ = float(self.values_.mean(axis=1).var() / spread) if spread > 0 else 0.0
        elif self.kind == 'metric':
            self._run_metric()
        elif self.kind == 'distance':
            self._run_distance()
        elif self.kind == 'invariant-density':
            bins = BINS['invariant-density'] if self.bins is None else self.bins
            self.density_, self.edges_ = np.histogram(self.values_, bins=bins, range=self.value_range, density=True)
        else:
            self._run_ulam()
