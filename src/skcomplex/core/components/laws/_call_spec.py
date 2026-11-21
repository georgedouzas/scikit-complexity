"""Define the call spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...base import BaseSpec
from ...errors import ValidationError


@dataclass(frozen=True)
class CallSpec(BaseSpec):
    """A spec of a law value that a Python function computes.

    It hands the value of a law to a function, such as `RateLaw(s.cell.n, CallSpec(growth, (s.cell.n,)))` inside an
    interaction, which calls `growth(t, n)` for the rate of every cell. The function runs once per member or pair and
    returns a number for a scalar target, or one number per entry for a vector target. The Numba backend calls a
    function that Numba compiles inside its own compiled loops, and the NumPy backend calls any function. A law that
    sets `relative`, `random`, or `vectorized` runs in Python on every backend. It reads a measured input from a table,
    asks a weather model for the rain on a crop, and runs a trained network for the force on a robot.

    Read more in the user guide, under Laws.

    Args:
        function:
            The function `f(t, *arguments)`, or `f(t, values)` when `arguments` is `None` or `relative` is true.

        arguments:
            The symbolic expressions whose values are passed to the function, or `None` to pass the mapping from the
            dotted name of every value and parameter to its value.

        relative:
            Whether to pass the mapping with the names of the component of the law, such as `x` for a value of the
            element itself and `target.x` for a value of the element bound to the role `target`. Those names do not
            depend on the label of the component or the system that holds it. It takes no `arguments`.

        random:
            Whether to pass the random generator of the run as the keyword argument `rng`, so that one seed repeats
            the run.

        vectorized:
            Whether to call the function once per evaluation with one array per argument, instead of once per member
            or pair. It returns one value, or one value per member or pair.

    See Also:
        RateLaw: Takes a call as its rate.
        EventLaw: Takes a call as the expression it watches.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import CallSpec, RateLaw
        >>> n = var('n')
        >>> RateLaw(n, CallSpec(lambda t, n: 2 * n, (n,))).value.arguments
        (n,)
    """

    function: object
    arguments: tuple[object, ...] | None = ()
    relative: bool = False
    random: bool = False
    vectorized: bool = False

    def _check(self) -> None:
        """Raise an error for a relative call with arguments, two ways to read values."""
        if self.relative and self.arguments:
            error_msg = (
                f'A CallSpec of {self.function!r} is relative and has the arguments {self.arguments!r}. A relative '
                'call reads its values by name, so remove the arguments or `relative=True`.'
            )
            raise ValidationError(error_msg)
