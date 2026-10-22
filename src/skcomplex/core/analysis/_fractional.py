"""Evaluate the Mittag-Leffler function, which solves linear fractional differential equations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from functools import cache
from typing import Any

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import ComplexField, RealField

from ..base import AnalysisError

ASYMPTOTIC_SCALE = 50.0
LARGEST_SCALE = 2000.0
GUARD_BITS = 64


def _reciprocal_gamma(argument: Any) -> Any:  # noqa: ANN401
    """Return `1 / Gamma(argument)` of a real number of a Sage field, which is 0 at the poles of the gamma function."""
    if argument <= 0 and argument == argument.round():
        return argument.parent()(0)
    return 1 / argument.gamma()


@cache
def _coefficients(alpha: float, beta: float, bits: int, count: int) -> tuple[Any, ...]:
    """Return the coefficients `1 / Gamma(alpha k + beta)` of the power series for `k < count` in a Sage field."""
    reals = RealField(bits)
    a, b = reals(alpha), reals(beta)
    return tuple(_reciprocal_gamma(a * index + b) for index in range(count))


def _series(alpha: float, beta: float, z: complex, scale: float) -> complex:
    """Return the Mittag-Leffler function by its power series, summed with enough precision for the cancellations."""
    extra = math.ceil(scale / math.log(2) + 2 * math.log2(1 + abs(z)))
    bits = 53 + GUARD_BITS * (1 + extra // GUARD_BITS)
    complexes = ComplexField(bits)
    w = complexes(z.real, z.imag)
    total, power = complexes(0), complexes(1)
    tolerance = RealField(bits)(2) ** -(53 + 10)
    count = 2 ** math.ceil(math.log2(scale / alpha + 16))
    coefficients = _coefficients(alpha, beta, bits, count)
    index = 0
    while True:
        if index == len(coefficients):
            coefficients = _coefficients(alpha, beta, bits, 2 * len(coefficients))
        term = power * coefficients[index]
        total += term
        if index > scale / alpha + 2 and abs(term) <= tolerance * abs(total):
            return complex(total)
        power *= w
        index += 1


def _asymptotic(alpha: float, beta: float, z: complex, scale: float) -> complex:
    """Return the Mittag-Leffler function by its asymptotic expansion for large `|z|`, with `0 < alpha < 2`.

    The expansion `E(z) = (1 / alpha) z^((1 - beta) / alpha) exp(z^(1 / alpha)) - sum_k z^(-k) / Gamma(beta - alpha k)`
    holds for `|arg z| < alpha pi`, and the exponential term drops out of it for larger arguments. The sum stops where
    the bound `|z|^(-k) Gamma(1 + alpha k - beta) / pi` of its terms falls below the precision or starts to grow.
    """
    reals, complexes = RealField(53 + GUARD_BITS), ComplexField(53 + GUARD_BITS)
    a, b, w = reals(alpha), reals(beta), complexes(z.real, z.imag)
    tolerance = reals(2) ** -(53 + 10)
    total = complexes(0)
    if abs(w.arg()) < a * reals.pi():
        total += w ** ((1 - b) / a) * (w ** (1 / a)).exp() / a
    for index in range(1, math.floor(scale) + 1):
        total -= w ** (-index) * _reciprocal_gamma(b - a * index)
        bound = abs(w) ** (-index) * (1 + a * index - b).gamma() / reals.pi()
        if bound <= tolerance * abs(total):
            break
    return complex(total)


def _evaluate(alpha: float, beta: float, z: complex) -> complex:
    """Return the Mittag-Leffler function at one point by its series or its asymptotic expansion."""
    if alpha == 1 and beta == 1:
        return complex(np.exp(z))
    scale = abs(z) ** (1 / alpha)
    if scale <= ASYMPTOTIC_SCALE:
        return _series(alpha, beta, z, scale)
    if alpha < 2:  # noqa: PLR2004
        return _asymptotic(alpha, beta, z, scale)
    if scale <= LARGEST_SCALE:
        return _series(alpha, beta, z, scale)
    error_msg = (
        f'E_(alpha, beta)(z) with alpha = {alpha} >= 2 is evaluated for |z|^(1/alpha) <= {LARGEST_SCALE}, but got '
        f'|z| = {abs(z)}. Rescale the argument.'
    )
    raise AnalysisError(error_msg)


def mittag_leffler(alpha: float, beta: float, z: npt.ArrayLike) -> npt.NDArray[Any]:
    """Return the two-parameter Mittag-Leffler function `E_(alpha, beta)(z) = sum_k z^k / Gamma(alpha k + beta)`.

    `E_(1, 1)(z)` is `exp(z)`, and `x_0 E_(alpha, 1)(-lambda t^alpha)` solves the fractional relaxation `D^alpha x =
    -lambda x` of a Caputo derivative of order `alpha`, which decays as the power law `t^(-alpha)` at long times. The
    function sums its power series in arbitrary precision with Sage for `|z|^(1/alpha) <= 50`, and takes its
    asymptotic expansion beyond.

    Args:
        alpha:
            The positive first parameter, the order of the derivative.

        beta:
            The real second parameter.

        z:
            The real or complex arguments.

    Returns:
        values:
            The values with the shape of `z`, real for real arguments.

    Raises:
        AnalysisError:
            `alpha` is not positive, or the argument is too large for `alpha >= 2`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import mittag_leffler
        >>> bool(np.isclose(mittag_leffler(1.0, 1.0, -2.0), np.exp(-2.0)))
        True
        >>> round(float(mittag_leffler(0.5, 1.0, -1.0)), 6)
        0.427584
    """
    if not alpha > 0:
        error_msg = f'alpha = {alpha!r} must be positive.'
        raise AnalysisError(error_msg)
    values = np.asarray(z)
    flat = [_evaluate(float(alpha), float(beta), complex(item)) for item in values.ravel()]
    result = np.asarray(flat, dtype=complex).reshape(values.shape)
    return result if np.iscomplexobj(values) else result.real
