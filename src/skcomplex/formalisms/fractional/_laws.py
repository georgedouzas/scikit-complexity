"""Define fractional noise, the amplitude of a fractional Brownian motion that drives a state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

from ...laws import Noise, Process, Target


@dataclass(frozen=True, init=False)
class FractionalNoise(Noise):
    """Amplitude of a fractional Brownian motion of a Hurst index that drives the highest time derivative of a state.

    The increments are fractional Gaussian noise, correlated over the whole run. A Hurst index of 1/2 gives the
    process of a `Noise` law, below 1/2 the increments anticorrelate, and above 1/2 they correlate. For other indices
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
            The stochastic calculus of the law, `'ito'`, `'stratonovich'`, or `'anti-ito'`, as for `Noise`.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.formalisms.fractional import FractionalNoise
        >>> x, D = symbol('x'), symbol('D')
        >>> FractionalNoise(x, 2 * D, hurst=0.7).hurst
        0.7
    """

    hurst: object = 0.5
    process: Process = field(default=None, init=False)
    conserved: bool = field(default=False, init=False)

    def __init__(
        self,
        target: Target,
        value: object,
        hurst: object,
        order: int | None = None,
        convention: str = 'ito',
    ) -> None:
        object.__setattr__(self, 'target', target)
        object.__setattr__(self, 'value', value)
        object.__setattr__(self, 'hurst', hurst)
        object.__setattr__(self, 'order', order)
        object.__setattr__(self, 'convention', convention)
        object.__setattr__(self, 'process', None)
        object.__setattr__(self, 'conserved', False)
        self.__post_init__()
