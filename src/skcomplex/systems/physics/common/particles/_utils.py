"""Define the helpers the particle realizations share."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from .....core.components.laws.utils import derivative


def velocity_of(position: object) -> Expression | FreeModuleElement:
    """Return the first time derivative of a position, component by component for a vector.

    Args:
        position:
            The symbol or the vector of symbols of a position.

    Returns:
        velocity:
            The derivative of order 1 of the position.
    """
    if isinstance(position, FreeModuleElement):
        return position.parent()([derivative(SR(item), 1) for item in position])
    return derivative(SR(position), 1)
