"""Define the splitting of complex components into their real and imaginary parts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

from sage.all__sagemath_symbolics import I

from ...core.base import Expression, components_of


def real_components_of(value: Any) -> list[Any]:  # noqa: ANN401
    """Return the components of a value with every complex component split into its real and imaginary parts.

    Args:
        value:
            A vector, an expression, or a number.

    Returns:
        components:
            The components, two in a row for a complex component.

    Examples:
        >>> from sage.all__sagemath_symbolics import I
        >>> from skcomplex.core.base import symbol, vector
        >>> from skcomplex.engine.compile._real_parts import real_components_of
        >>> x, y = symbol('x'), symbol('y')
        >>> real_components_of(vector([x + I * y, 2 * x]))
        [x, y, 2*x]
    """
    components = []
    for item in components_of(value):
        if isinstance(item, Expression) and item.has(I):
            components += [item.real_part(), item.imag_part()]
        else:
            components.append(item)
    return components
