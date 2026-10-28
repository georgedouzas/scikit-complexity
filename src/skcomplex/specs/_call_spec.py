"""Define the call spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class CallSpec(BaseSpec):
    """Law value computed by a function of time and scalar arguments, called once per member or pair.

    The Numba backend calls a Numba-compiled function inside its compiled kernels, and the NumPy backend calls any
    function from Python. The function returns a number for a scalar target, or one number per component for a vector
    target.

    Args:
        function:
            The function `f(t, *arguments)`.

        arguments:
            The symbolic expressions whose values are passed to the function.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.specs import CallSpec
        >>> n = var('n')
        >>> CallSpec(lambda t, n: 2 * n, (n,)).arguments
        (n,)
    """

    function: object
    arguments: tuple[object, ...] = ()
