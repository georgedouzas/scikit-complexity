"""Define fractional noise laws.

They are amplitudes of the fractional Brownian motions that drive states.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...errors import ValidationError
from ..types import Target
from ._noise import NoiseLaw


class FractionalNoiseLaw(NoiseLaw):
    """Amplitude of a fractional Brownian motion of a Hurst index.

    The motion drives the highest time derivative of a state. The increments are fractional Gaussian noise, correlated
    over the whole run. A Hurst index of 1/2 gives the process of a `NoiseLaw`, below 1/2 the increments anticorrelate,
    and above 1/2 they correlate. For other indices than 1/2 the amplitude must not depend on any state.

    Args:
        target:
            The symbol of a scalar state, or the column of symbols of a vector state.

        value:
            A symbolic expression of parameters and time. A vector target gets one fractional Brownian motion per
            component.

        order:
            The order of the derivative the noise drives, checked against the state's order, or `None` to use it.

        hurst:
            The Hurst index `0 < H < 1`, a number or a parameter.

        convention:
            The stochastic calculus of the law, `'ito'`, `'stratonovich'`, or `'anti-ito'`, as for `NoiseLaw`.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import FractionalNoiseLaw
        >>> x, D = var('x'), var('D')
        >>> FractionalNoiseLaw(x, 2 * D, hurst=0.7).hurst
        0.7
    """

    def __init__(
        self,
        target: Target,
        value: object,
        order: int | None = None,
        *,
        hurst: object,
        convention: str = 'ito',
        label: str = '',
    ) -> None:
        super().__init__(target, value, order, convention=convention, label=label)
        self.hurst = hurst

    def _prepare(self, context: str | None = None) -> None:
        """Raise an error for an invalid noise or a numeric Hurst index outside `(0, 1)`.

        Args:
            context:
                The path of the element or interaction that returns the law, or `None`.
        """
        super()._prepare(context)
        if isinstance(self.hurst, int | float) and not 0 < self.hurst < 1:
            error_msg = f'The Hurst index of the noise of {self.target} is {self.hurst}. Give a number in (0, 1).'
            raise ValidationError(error_msg)
