"""Define the call spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class CallSpec(BaseSpec):
    """Law value computed by a function of time and scalar arguments.

    The function is called once per member or pair. The Numba backend calls a Numba-compiled function inside its
    compiled kernels, and the NumPy backend calls any function from Python. The function returns a number for a scalar
    target, or one number per component for a vector target. A law that sets `relative`, `random`, or `vectorized` runs
    in Python on every backend.

    Args:
        function:
            The function `f(t, *arguments)`, or `f(t, values)` when `arguments` is `None` or `relative` is true.

        arguments:
            The symbolic expressions whose values are passed to the function, or `None` to pass the mapping from the
            dotted name of every state and parameter to its value.

        relative:
            Whether to pass the mapping with the names of the component of the law, such as `x` for a state of the
            element itself and `target.x` for a state of the element bound to the role `target`. Those names do not
            depend on the label of the component or the system that holds it. It takes no `arguments`.

        random:
            Whether to pass the random stream of the simulation as the keyword argument `rng`, so that one seed repeats
            the run.

        vectorized:
            Whether to call the function once per evaluation with one array per argument, instead of once per member
            or pair. It returns one value, or one value per member or pair.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.specs import CallSpec
        >>> n = var('n')
        >>> CallSpec(lambda t, n: 2 * n, (n,)).arguments
        (n,)
        >>> CallSpec(lambda t, values: values['n'], relative=True).relative
        True
    """

    function: object
    arguments: tuple[object, ...] | None = ()
    relative: bool = False
    random: bool = False
    vectorized: bool = False

    def _check(self) -> None:
        """Raise an error for relative names next to arguments.

        They are two ways to read values.
        """
        if self.relative and self.arguments:
            error_msg = (
                f'A CallSpec of {self.function!r} is relative and has the arguments {self.arguments!r}. A relative '
                'call reads its values by name, so remove the arguments or `relative=True`.'
            )
            raise ValidationError(error_msg)
