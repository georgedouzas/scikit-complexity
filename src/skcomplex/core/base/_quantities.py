"""Define the base of the quantities, the named expressions of the states and parameters of a system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True)
class BaseQuantity:
    """Base of every quantity, a frozen dataclass of a label and an expression of states and parameters.

    A quantity is either an expression of the symbols of the component or the system that declares it, or a reduction
    over the members of a group. Its children differ in what the framework does with it, such as recording it,
    substituting it into laws, checking that it stays constant, optimizing it, or giving it to other partitions.

    Args:
        label:
            The name of the quantity.

        expression:
            The expression of states and parameters, or a function of a namespace or of a member.

    Examples:
        >>> from skcomplex.core.base import BaseQuantity, BaseLaw, symbol
        >>> from skcomplex.core.laws import AuxiliaryQuantity
        >>> births = AuxiliaryQuantity('births', symbol('b') * symbol('n'))
        >>> isinstance(births, BaseQuantity), isinstance(births, BaseLaw)
        (True, False)
        >>> births.check().label
        'births'
    """

    label: str
    expression: object

    def _check(self) -> Self:
        """Return the quantity with its values checked and normalized, the quantity itself by default."""
        return self

    def check(self) -> Self:
        """Return the quantity with its values checked and normalized.

        Quantities are values, so checking returns a quantity rather than changing this one. Validation checks every
        quantity of a system before compiling it.

        Returns:
            quantity:
                The checked quantity.

        Raises:
            ValidationError:
                A value of the quantity is invalid, such as a NumPy array or a lambda that cannot be pickled.
        """
        return self._check()
