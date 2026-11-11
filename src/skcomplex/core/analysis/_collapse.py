"""Measure collapses of curves."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize

from ..enums import DataKind
from ..errors import AnalysisError
from ..specs import AnalysisTags
from ..types import Array
from ._base import _check_choice, _check_settings
from .types import Source

KINDS = ('scaling', 'aging', 'distribution', 'crossings')
FITTED = ('critical', 'nu', 'exponent')
SETTINGS = {
    'scaling': {'parameters', 'critical', 'nu', 'exponent', 'fit'},
    'aging': {'waiting', 'lags'},
    'distribution': {'ratios'},
    'crossings': {'parameters'},
}
MIN_SIZES = 2
GRID = 50
TAGS = AnalysisTags(DataKind.ENSEMBLE)


@dataclass(frozen=True)
class DataCollapseResult:
    """A result of `data_collapse`, curves of several sizes collapsed onto one.

    Args:
        values:
            The curve of every size, the curves of every waiting time, or the samples of every age.

        labels:
            The size, waiting time, or age of every curve.

        critical:
            The critical value used or fitted, for `'scaling'`.

        nu:
            The correlation length exponent used or fitted, for `'scaling'`.

        exponent:
            The exponent `a` used or fitted, for `'scaling'`.

        quality:
            The mean squared relative distance between the scaled curves, 0 for a perfect collapse, for `'scaling'`.

        time_spread:
            The largest difference between the curves against the lag, 0 without aging, for `'aging'`.

        ratio_spread:
            The largest difference between the curves against `tau / t_w`, 0 under simple aging, for `'aging'`.

        ratios:
            The centres of the bins of `tau / t`, for `'distribution'`.

        densities:
            One row per age of `t P(tau, t)` on the bins, for `'distribution'`.

        spread:
            The largest relative difference between the rows where every row is positive, for `'distribution'`.

        crossings:
            The first crossing of the curves of every pair of consecutive sizes, `nan` where they do not cross, for
            `'crossings'`.
    """

    values: dict[float, Array] | Array
    labels: list[float]
    critical: float | None = None
    nu: float | None = None
    exponent: float | None = None
    quality: float | None = None
    time_spread: float | None = None
    ratio_spread: float | None = None
    ratios: Array | None = None
    densities: Array | None = None
    spread: float | None = None
    crossings: Array | None = None


def data_collapse(
    source: Source,
    *,
    kind: str = 'scaling',
    parameters: Sequence[float] | None = None,
    critical: float | None = None,
    nu: float | None = None,
    exponent: float = 0.0,
    fit: Sequence[str] = (),
    waiting: Sequence[float] | None = None,
    lags: Sequence[float] | None = None,
    ratios: Sequence[float] | None = None,
) -> DataCollapseResult:
    """Return the collapse of curves or distributions of several sizes or ages.

    The curves or distributions collapse onto one, of one kind. The collapse is of one kind.

    The kinds read the curves or the samples of every size or age:

    - `'scaling'`: the finite-size scaling collapse of curves `y_L(p) = L^a f((p - p_c) L^(1/nu))` of every size
      `L`, against the control `parameters`, with the names of `fit` among `'critical'`, `'nu'`, and `'exponent'`
      fitted by minimizing the quality.
    - `'aging'`: how far correlation curves of several waiting times stay apart against the lag and against
      `tau / t_w`, from one curve per waiting time, as the two-time correlation of `correlation` gives them.
    - `'distribution'`: the densities `t P(tau, t)` of samples of `tau` at several ages `t` against `tau / t`, which
      collapse under aging.
    - `'crossings'`: where the curves of consecutive sizes cross against the control `parameters`, by linear
      interpolation of their difference, such as the Binder cumulants of several lattice sizes at the critical point.

    It appears in many fields. It estimates the critical exponents of a magnet from lattices of several sizes, and the
    growth exponents of a rough surface.

    Args:
        source:
            A mapping from size or age to its curve or samples, or an array with one curve per row.

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

    Returns:
        result:
            The curves, their labels, and the collapse of the kind.

    Raises:
        AnalysisError:
            The kind or a fitted name is unknown, a setting the kind needs is missing or one it does not read is
            changed, or the source has too few sizes or is not a mapping where the kind needs one.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import data_collapse
        >>> grid = np.linspace(-1, 1, 401)
        >>> curves = {size: np.tanh(grid * size) for size in (8, 16, 32)}
        >>> result = data_collapse(curves, parameters=grid, critical=0.2, nu=1.5, fit=('critical', 'nu'))
        >>> round(result.nu, 1)
        1.0
        >>> lags = np.linspace(1, 100, 100)
        >>> curves = [np.exp(-lags / tw) for tw in (10, 100)]
        >>> aging = data_collapse(curves, kind='aging', waiting=[10, 100], lags=lags)
        >>> round(aging.ratio_spread, 2), aging.time_spread > 0.5
        (0.0, True)
        >>> curves = {8: [0.6, 0.5, 0.2], 16: [0.65, 0.5, 0.1]}
        >>> data_collapse(curves, kind='crossings', parameters=[1.0, 2.0, 3.0]).crossings.tolist()
        [2.0]
    """
    _check_choice('data_collapse', 'kind', kind, KINDS)
    for name in fit:
        _check_choice('data_collapse', 'fit', name, FITTED)
    _check_settings(
        data_collapse,
        'kind',
        SETTINGS[kind],
        kind=kind,
        parameters=parameters,
        critical=critical,
        nu=nu,
        exponent=exponent,
        fit=fit,
        waiting=waiting,
        lags=lags,
        ratios=ratios,
    )
    if kind == 'scaling' and (parameters is None or critical is None or nu is None):
        error_msg = 'data_collapse of kind scaling needs parameters, critical, and nu. Give all three.'
        raise AnalysisError(error_msg)
    if kind == 'crossings' and parameters is None:
        error_msg = 'data_collapse of kind crossings needs the control parameter values. Set `parameters`.'
        raise AnalysisError(error_msg)
    if kind == 'aging' and (waiting is None or lags is None):
        error_msg = 'data_collapse of kind aging needs waiting times and lags. Give both.'
        raise AnalysisError(error_msg)
    values: dict[float, Array] | Array
    if isinstance(source, Mapping):
        values = {key: np.asarray(value, dtype=np.float64) for key, value in source.items()}
        labels = [float(key) for key in source]
    else:
        values = np.asarray(source, dtype=np.float64)
        labels = [float(item) for item in waiting] if waiting is not None else list(range(len(values)))
    if kind in {'scaling', 'crossings'} and len(values) < MIN_SIZES:
        error_msg = f'data_collapse needs at least {MIN_SIZES} sizes, not {len(values)}. Give more sizes.'
        raise AnalysisError(error_msg)
    if kind == 'scaling' and critical is not None and nu is not None:
        start = {'critical': critical, 'nu': nu, 'exponent': exponent}
        grid = np.asarray(parameters, dtype=np.float64)
        fitted, quality = _scaling_collapse(grid, _curves_by_key(values, kind), start, fit)
        return DataCollapseResult(
            values,
            labels,
            critical=fitted['critical'],
            nu=fitted['nu'],
            exponent=fitted['exponent'],
            quality=quality,
        )
    if kind == 'aging' and waiting is not None:
        grid = np.asarray(lags, dtype=np.float64)
        curves = np.asarray(values, dtype=float)
        return DataCollapseResult(
            values,
            labels,
            time_spread=_spread([(grid, row) for row in curves]),
            ratio_spread=_spread([(grid / start, row) for start, row in zip(waiting, curves, strict=True)]),
        )
    if kind == 'crossings':
        rows = np.asarray([values[size] for size in sorted(values)]) if isinstance(values, dict) else values
        return DataCollapseResult(values, labels, crossings=_crossings(np.asarray(parameters, dtype=np.float64), rows))
    edges = np.geomspace(0.01, 10, 21) if ratios is None else np.asarray(ratios, dtype=float)
    densities, spread = _scaled_densities(_curves_by_key(values, kind), edges)
    return DataCollapseResult(
        values,
        labels,
        ratios=np.sqrt(edges[1:] * edges[:-1]),
        densities=densities,
        spread=spread,
    )


def _curves_by_key(values: dict[float, Array] | Array, kind: str) -> dict[float, Array]:
    """Return the curves or the samples by size or age.

    The kinds scaling and distribution read them.
    """
    if not isinstance(values, dict):
        error_msg = f'data_collapse of kind {kind} reads a mapping from size or age to its values.'
        raise AnalysisError(error_msg)
    return values


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


def _quality(scaled: list[tuple[Array, Array]]) -> float:
    """Return the mean squared relative distance of every point to the other curves.

    The distance is taken where they overlap.
    """
    residuals = []
    for (x, y), (u, v) in itertools.permutations(scaled, 2):
        inside = (x >= u.min()) & (x <= u.max())
        residuals.append((y[inside] - np.interp(x[inside], u, v)) ** 2)
    joined = np.concatenate(residuals) if residuals else np.zeros(0)
    if not joined.size:
        return float('inf')
    scale = np.mean(np.concatenate([y for _, y in scaled]) ** 2)
    return float(joined.mean() / scale)


def _scaling_collapse(
    grid: Array,
    curves: Mapping[float, Array],
    values: dict[str, float],
    fit: Sequence[str],
) -> tuple[dict[str, float], float]:
    """Return the critical value, `nu`, the exponent, and the quality.

    The values are fitted by minimizing the quality.
    """

    def objective(free: Array) -> float:
        """Return the quality with the fitted values replaced."""
        chosen = {**values, **dict(zip(fit, free, strict=True))}
        if chosen['nu'] <= 0:
            return float('inf')
        return _quality(
            _scaled(grid, curves, chosen['critical'], chosen['nu'], chosen['exponent']),
        )

    values = dict(values)
    if fit:
        result = optimize.minimize(objective, [values[name] for name in fit], method='Nelder-Mead')
        values.update(dict(zip(fit, (float(item) for item in result.x), strict=True)))
    return values, _quality(
        _scaled(grid, curves, values['critical'], values['nu'], values['exponent']),
    )


def _spread(curves: list[tuple[Array, Array]]) -> float:
    """Return the largest difference between curves interpolated on their overlap.

    The overlap is that of their abscissas.
    """
    low = max(float(x.min()) for x, _ in curves)
    high = min(float(x.max()) for x, _ in curves)
    if low >= high:
        return float('nan')
    grid = np.linspace(low, high, GRID)
    values = np.asarray([np.interp(grid, x, y) for x, y in curves])
    return float(np.max(values.max(axis=0) - values.min(axis=0)))


def _scaled_densities(samples: Mapping[float, Array], edges: Array) -> tuple[Array, float]:
    """Return the densities `t P(tau, t)` of samples at every age against `tau / t`.

    Their largest spread is returned too.
    """
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


def _crossings(grid: Array, curves: Array) -> Array:
    """Return the first crossing of every pair of consecutive curves.

    There is one per row, and `nan` without one.
    """
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
