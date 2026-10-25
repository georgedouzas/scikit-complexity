"""Define fractional noise, the amplitude of a fractional Brownian motion that drives a state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

from ...core.laws import NoiseLaw, Process


@dataclass(frozen=True)
class FractionalNoiseLaw(NoiseLaw):
    """Amplitude of a fractional Brownian motion of a Hurst index that drives the highest time derivative of a state.

    The increments are fractional Gaussian noise, correlated over the whole run. A Hurst index of 1/2 gives the
    process of a `NoiseLaw`, below 1/2 the increments anticorrelate, and above 1/2 they correlate. For other indices
    than 1/2 the amplitude must not depend on any state.

    Args:
        target:
            The symbol of a scalar state, or the column of symbols of a vector state.

        value:
            A symbolic expression of parameters and time. A vector target gets one fractional Brownian motion per
            component.

        hurst:
            The Hurst index `0 < H < 1`, a number or a parameter.

        order:
            The order of the derivative the noise drives, checked against the state's order, or `None` to use it.

        convention:
            The stochastic calculus of the law, `'ito'`, `'stratonovich'`, or `'anti-ito'`, as for `NoiseLaw`.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.formalisms.fractional import FractionalNoiseLaw
        >>> x, D = symbol('x'), symbol('D')
        >>> FractionalNoiseLaw(x, 2 * D, hurst=0.7).hurst
        0.7
    """

    process: Process = field(default=None, init=False)
    conserved: bool = field(default=False, init=False)
    hurst: object = field(kw_only=True)
