"""Locate transitions by finite-size scaling: Binder cumulants, their crossings, and scaling collapse."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import optimize

from ..base import AnalysisError
from ._kinds import DataKind, analysis

type Array = npt.NDArray[np.float64]
MIN_SIZES = 2


@analysis(DataKind.FIELD)
def binder_cumulant(samples: npt.ArrayLike) -> float:
    """Return the Binder cumulant `1 - <m^4> / (3 <m^2>^2)` of samples of an order parameter.

    Args:
        samples:
            The samples of the order parameter.

    Returns:
        cumulant:
            2/3 deep in an ordered phase and 0 for Gaussian fluctuations about 0.

    Examples:
        >>> from skcomplex.core.analysis import binder_cumulant
        >>> round(binder_cumulant([1.0, -1.0, 1.0]), 4)
        0.6667
    """
    array = np.asarray(samples, dtype=float)
    return float(1 - np.mean(array**4) / (3 * np.mean(array**2) ** 2))


@analysis(DataKind.ENSEMBLE)
def binder_crossings(parameters: npt.ArrayLike, cumulants: Mapping[int, npt.ArrayLike]) -> Array:
    """Return where the Binder cumulants of consecutive sizes cross, by linear interpolation of their difference.

    Args:
        parameters:
            The control parameter values, such as temperatures, shared by every size.

        cumulants:
            The cumulants at every parameter for every size.

    Returns:
        crossings:
            The first crossing of every pair of consecutive sizes, `nan` where they do not cross.

    Raises:
        AnalysisError:
            Fewer than 2 sizes are given.

    Examples:
        >>> from skcomplex.core.analysis import binder_crossings
        >>> binder_crossings([1.0, 2.0, 3.0], {8: [0.6, 0.5, 0.2], 16: [0.65, 0.5, 0.1]}).tolist()
        [2.0]
    """
    sizes = sorted(cumulants)
    if len(sizes) < MIN_SIZES:
        error_msg = f'Crossings need at least {MIN_SIZES} sizes, not {len(sizes)}.'
        raise AnalysisError(error_msg)
    grid = np.asarray(parameters, dtype=float)
    crossings = []
    for small, large in itertools.pairwise(sizes):
        difference = np.asarray(cumulants[large], dtype=float) - np.asarray(cumulants[small], dtype=float)
        changes = np.flatnonzero(np.sign(difference[:-1]) * np.sign(difference[1:]) <= 0)
        if not changes.size:
            crossings.append(np.nan)
            continue
        index = int(changes[0])
        left, right = difference[index], difference[index + 1]
        share = left / (left - right) if left != right else 0.0
        crossings.append(grid[index] + share * (grid[index + 1] - grid[index]))
    return np.asarray(crossings, dtype=float)


def _scaled(
    grid: Array,
    curves: Mapping[int, Array],
    critical: float,
    nu: float,
    exponent: float,
) -> list[tuple[Array, Array]]:
    """Return every curve against `(p - p_c) L^(1/nu)` and scaled by `L^-exponent`."""
    return [
        ((grid - critical) * size ** (1 / nu), curve * float(size) ** (-exponent)) for size, curve in curves.items()
    ]


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


@dataclass(frozen=True)
class Collapse:
    """Finite-size scaling collapse of curves of several sizes.

    Args:
        critical:
            The critical value of the control parameter.

        nu:
            The correlation length exponent.

        exponent:
            The exponent `a` of the scaled curves `y L^-a`, such as `gamma / nu` for a susceptibility.

        quality:
            The mean squared relative distance between the scaled curves, 0 for a perfect collapse.
    """

    critical: float
    nu: float
    exponent: float
    quality: float


@analysis(DataKind.ENSEMBLE)
def scaling_collapse(
    parameters: npt.ArrayLike,
    curves: Mapping[int, npt.ArrayLike],
    *,
    critical: float,
    nu: float,
    exponent: float = 0.0,
    fit: tuple[str, ...] = (),
) -> Collapse:
    """Return the finite-size scaling collapse of curves `y_L(p) = L^a f((p - p_c) L^(1/nu))` and its quality.

    Args:
        parameters:
            The control parameter values, shared by every size.

        curves:
            The curve of every size.

        critical:
            The critical value, or the starting value of a fit.

        nu:
            The correlation length exponent, or the starting value of a fit.

        exponent:
            The exponent `a`, or the starting value of a fit.

        fit:
            The names among `'critical'`, `'nu'`, and `'exponent'` to fit by minimizing the quality.

    Returns:
        collapse:
            The values used or fitted and the quality.

    Raises:
        AnalysisError:
            Fewer than 2 sizes are given.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import scaling_collapse
        >>> grid = np.linspace(-1, 1, 401)
        >>> curves = {size: np.tanh(grid * size) for size in (8, 16, 32)}
        >>> round(scaling_collapse(grid, curves, critical=0.2, nu=1.5, fit=('critical', 'nu')).nu, 1)
        1.0
    """
    if len(curves) < MIN_SIZES:
        error_msg = f'A collapse needs at least {MIN_SIZES} sizes, not {len(curves)}.'
        raise AnalysisError(error_msg)
    grid = np.asarray(parameters, dtype=float)
    arrays = {size: np.asarray(curve, dtype=float) for size, curve in curves.items()}
    values = {'critical': critical, 'nu': nu, 'exponent': exponent}

    def objective(free: Array) -> float:
        """Return the quality with the fitted values replaced."""
        chosen = {**values, **dict(zip(fit, free, strict=True))}
        if chosen['nu'] <= 0:
            return float('inf')
        return _quality(_scaled(grid, arrays, chosen['critical'], chosen['nu'], chosen['exponent']))

    if fit:
        result = optimize.minimize(objective, [values[name] for name in fit], method='Nelder-Mead')
        values.update(dict(zip(fit, (float(item) for item in result.x), strict=True)))
    quality = _quality(_scaled(grid, arrays, values['critical'], values['nu'], values['exponent']))
    return Collapse(values['critical'], values['nu'], values['exponent'], quality)
