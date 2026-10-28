"""Measure collapses of curves: finite-size scaling, aging, distributions scaled by age, and crossings of curves."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Mapping, Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy import optimize

from ..core.errors import AnalysisError
from ._base import BaseAnalysis, Source
from ._kinds import DataKind

type Array = npt.NDArray[np.float64]
KINDS = ('scaling', 'aging', 'distribution', 'crossings')
FITTED = ('critical', 'nu', 'exponent')
SETTINGS = {
    'scaling': {'parameters', 'critical', 'nu', 'exponent', 'fit'},
    'aging': {'waiting', 'lags'},
    'distribution': {'ratios'},
    'crossings': {'parameters'},
}
"""Settings every kind reads, besides `kind`."""
MIN_SIZES = 2
GRID = 50


class CollapseAnalysis(BaseAnalysis):
    """Collapse of curves or distributions of several sizes or ages onto one, of one kind.

    The kinds read the curves or the samples of every size or age:

    - `'scaling'`: the finite-size scaling collapse of curves `y_L(p) = L^a f((p - p_c) L^(1/nu))` of every size
      `L`, against the control `parameters`, with the names of `fit` among `'critical'`, `'nu'`, and `'exponent'`
      fitted by minimizing the quality.
    - `'aging'`: how far correlation curves of several waiting times stay apart against the lag and against
      `tau / t_w`, from one curve per waiting time, as the two-time correlation of `CorrelationAnalysis` gives them.
    - `'distribution'`: the densities `t P(tau, t)` of samples of `tau` at several ages `t` against `tau / t`, which
      collapse under aging.
    - `'crossings'`: where the curves of consecutive sizes cross against the control `parameters`, by linear
      interpolation of their difference, such as the Binder cumulants of several lattice sizes at the critical point.

    Args:
        kind:
            The collapse to compute, one of the kinds above.

        parameters:
            The control parameter values, shared by every curve of `'scaling'` and `'crossings'`.

        critical:
            The critical value, or the starting value of a fit.

        nu:
            The correlation length exponent, or the starting value of a fit.

        exponent:
            The exponent `a` of the scaled curves `y L^-a`, such as `gamma / nu` for a susceptibility, or the
            starting value of a fit.

        fit:
            The names among `'critical'`, `'nu'`, and `'exponent'` to fit.

        waiting:
            The waiting times of the curves of `'aging'`.

        lags:
            The lags shared by every curve of `'aging'`.

        ratios:
            The edges of the bins of `tau / t`, or `None` for 20 logarithmic bins from 0.01 to 10.

    Attributes:
        values_:
            The curve of every size, the curves of every waiting time, or the samples of every age.

        labels_:
            The size, waiting time, or age of every curve.

        critical_:
            The critical value used or fitted.

        nu_:
            The correlation length exponent used or fitted.

        exponent_:
            The exponent `a` used or fitted.

        quality_:
            The mean squared relative distance between the scaled curves, 0 for a perfect collapse.

        time_spread_:
            The largest difference between the curves against the lag, 0 without aging.

        ratio_spread_:
            The largest difference between the curves against `tau / t_w`, 0 under simple aging.

        ratios_:
            The centres of the bins of `tau / t`.

        densities_:
            One row per age of `t P(tau, t)` on the bins.

        spread_:
            The largest relative difference between the rows where every row is positive.

        crossings_:
            The first crossing of the curves of every pair of consecutive sizes, `nan` where they do not cross.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import CollapseAnalysis
        >>> grid = np.linspace(-1, 1, 401)
        >>> curves = {size: np.tanh(grid * size) for size in (8, 16, 32)}
        >>> collapse = CollapseAnalysis(parameters=grid, critical=0.2, nu=1.5, fit=('critical', 'nu'))
        >>> round(collapse.check().prepare(curves).run().nu_, 1)
        1.0
        >>> lags = np.linspace(1, 100, 100)
        >>> curves = [np.exp(-lags / tw) for tw in (10, 100)]
        >>> aging = CollapseAnalysis('aging', waiting=[10, 100], lags=lags).check().prepare(curves).run()
        >>> round(aging.ratio_spread_, 2), aging.time_spread_ > 0.5
        (0.0, True)
        >>> curves = {8: [0.6, 0.5, 0.2], 16: [0.65, 0.5, 0.1]}
        >>> CollapseAnalysis('crossings', parameters=[1.0, 2.0, 3.0]).check().prepare(curves).run().crossings_.tolist()
        [2.0]
    """

    def __init__(
        self,
        kind: str = 'scaling',
        *,
        parameters: Sequence[float] | None = None,
        critical: float | None = None,
        nu: float | None = None,
        exponent: float = 0.0,
        fit: Sequence[str] = (),
        waiting: Sequence[float] | None = None,
        lags: Sequence[float] | None = None,
        ratios: Sequence[float] | None = None,
    ) -> None:
        self.kind = kind
        self.parameters = parameters
        self.critical = critical
        self.nu = nu
        self.exponent = exponent
        self.fit = fit
        self.waiting = waiting
        self.lags = lags
        self.ratios = ratios

    def _check(self) -> Self:
        """Validate the kind, the fitted names, the settings the kind needs, and that it reads every changed one."""
        self._check_choice('kind', self.kind, KINDS)
        for name in self.fit:
            self._check_choice('fit', name, FITTED)
        self._check_settings('kind', SETTINGS[self.kind])
        if self.kind == 'scaling' and (self.parameters is None or self.critical is None or self.nu is None):
            error_msg = f'{type(self).__name__} of kind scaling needs parameters, critical, and nu. Give all three.'
            raise AnalysisError(error_msg)
        if self.kind == 'crossings' and self.parameters is None:
            error_msg = f'{type(self).__name__} of kind crossings needs the control parameter values. Set `parameters`.'
            raise AnalysisError(error_msg)
        if self.kind == 'aging' and (self.waiting is None or self.lags is None):
            error_msg = f'{type(self).__name__} of kind aging needs waiting times and lags. Give both.'
            raise AnalysisError(error_msg)
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the ensemble kind, since every curve or sample comes from a run or a set of runs."""
        return DataKind.ENSEMBLE

    def _prepare(self, source: Source) -> Self:
        """Read the curves or the samples of every size or age, or the rows of an array of curves."""
        if isinstance(source, Mapping):
            self.values_ = {key: np.asarray(value, dtype=float) for key, value in source.items()}
            self.labels_ = list(source)
            return self
        self.values_ = np.asarray(source, dtype=float)
        self.labels_ = list(self.waiting) if self.waiting is not None else list(range(len(self.values_)))
        return self

    def _run(self, inputs: object = None) -> Self:
        """Compute the collapse of the kind."""
        if self.kind in {'scaling', 'crossings'} and len(self.values_) < MIN_SIZES:
            error_msg = (
                f'{type(self).__name__} needs at least {MIN_SIZES} sizes, not {len(self.values_)}. Give more sizes.'
            )
            raise AnalysisError(error_msg)
        if self.kind == 'scaling':
            start = {'critical': self.critical, 'nu': self.nu, 'exponent': self.exponent}
            grid = np.asarray(self.parameters, dtype=float)
            values, self.quality_ = self._scaling_collapse(grid, self.values_, start, self.fit)
            self.critical_, self.nu_, self.exponent_ = values['critical'], values['nu'], values['exponent']
        elif self.kind == 'aging':
            grid = np.asarray(self.lags, dtype=float)
            curves = np.asarray(self.values_, dtype=float)
            self.time_spread_ = self._spread([(grid, row) for row in curves])
            self.ratio_spread_ = self._spread(
                [(grid / start, row) for start, row in zip(self.waiting, curves, strict=True)],
            )
        elif self.kind == 'crossings':
            values = self.values_
            curves = [values[size] for size in sorted(values)] if isinstance(values, dict) else values
            self.crossings_ = self._crossings(np.asarray(self.parameters, dtype=float), np.asarray(curves, dtype=float))
        else:
            edges = np.geomspace(0.01, 10, 21) if self.ratios is None else np.asarray(self.ratios, dtype=float)
            self.densities_, self.spread_ = self._scaled_densities(self.values_, edges)
            self.ratios_ = np.sqrt(edges[1:] * edges[:-1])
        return self

    @staticmethod
    def _scaled(
        grid: Array,
        curves: Mapping[float, Array],
        critical: float,
        nu: float,
        exponent: float,
    ) -> list[tuple[Array, Array]]:
        """Return every curve against `(p - p_c) L^(1/nu)` and scaled by `L^-exponent`."""
        return [
            ((grid - critical) * size ** (1 / nu), curve * float(size) ** (-exponent)) for size, curve in curves.items()
        ]

    @staticmethod
    def _quality(scaled: list[tuple[Array, Array]]) -> float:
        """Return the mean squared relative distance of every point to the other curves where they overlap."""
        residuals = []
        for (x, y), (u, v) in itertools.permutations(scaled, 2):
            inside = (x >= u.min()) & (x <= u.max())
            residuals.append((y[inside] - np.interp(x[inside], u, v)) ** 2)
        joined = np.concatenate(residuals) if residuals else np.zeros(0)
        if not joined.size:
            return float('inf')
        scale = np.mean(np.concatenate([y for _, y in scaled]) ** 2)
        return float(joined.mean() / scale)

    @staticmethod
    def _scaling_collapse(
        grid: Array,
        curves: Mapping[float, Array],
        values: dict[str, float],
        fit: Sequence[str],
    ) -> tuple[dict[str, float], float]:
        """Return the critical value, `nu`, and the exponent, fitted by minimizing the quality, and the quality."""

        def objective(free: Array) -> float:
            """Return the quality with the fitted values replaced."""
            chosen = {**values, **dict(zip(fit, free, strict=True))}
            if chosen['nu'] <= 0:
                return float('inf')
            return CollapseAnalysis._quality(
                CollapseAnalysis._scaled(grid, curves, chosen['critical'], chosen['nu'], chosen['exponent']),
            )

        values = dict(values)
        if fit:
            result = optimize.minimize(objective, [values[name] for name in fit], method='Nelder-Mead')
            values.update(dict(zip(fit, (float(item) for item in result.x), strict=True)))
        return values, CollapseAnalysis._quality(
            CollapseAnalysis._scaled(grid, curves, values['critical'], values['nu'], values['exponent']),
        )

    @staticmethod
    def _spread(curves: list[tuple[Array, Array]]) -> float:
        """Return the largest difference between curves interpolated on the overlap of their abscissas."""
        low = max(float(x.min()) for x, _ in curves)
        high = min(float(x.max()) for x, _ in curves)
        if low >= high:
            return float('nan')
        grid = np.linspace(low, high, GRID)
        values = np.asarray([np.interp(grid, x, y) for x, y in curves])
        return float(np.max(values.max(axis=0) - values.min(axis=0)))

    @staticmethod
    def _scaled_densities(samples: Mapping[float, Array], edges: Array) -> tuple[Array, float]:
        """Return the densities `t P(tau, t)` of samples at every age against `tau / t`, and their largest spread."""
        ages = sorted(samples)
        densities = np.asarray(
            [np.histogram(samples[age] / age, bins=edges)[0] / (samples[age].size * np.diff(edges)) for age in ages],
        )
        positive = np.all(densities > 0, axis=0)
        spread = float('nan')
        if positive.any():
            kept = densities[:, positive]
            spread = float(np.max((kept.max(axis=0) - kept.min(axis=0)) / kept.mean(axis=0)))
        return densities, spread

    @staticmethod
    def _crossings(grid: Array, curves: Array) -> Array:
        """Return the first crossing of every pair of consecutive curves, one per row, `nan` without one."""
        crossings = []
        for small, large in itertools.pairwise(curves):
            difference = large - small
            changes = np.flatnonzero(np.sign(difference[:-1]) * np.sign(difference[1:]) <= 0)
            if not changes.size:
                crossings.append(np.nan)
                continue
            index = int(changes[0])
            left, right = difference[index], difference[index + 1]
            share = left / (left - right) if left != right else 0.0
            crossings.append(grid[index] + share * (grid[index + 1] - grid[index]))
        return np.asarray(crossings, dtype=float)
