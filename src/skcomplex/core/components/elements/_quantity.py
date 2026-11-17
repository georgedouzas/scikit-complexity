"""Define the quantity a value of an element holds."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...errors import ValidationError
from ...specs._base_spec import BaseSpec


@dataclass(frozen=True)
class Quantity(BaseSpec):
    """A conserved quantity that a value of an element holds.

    It is written in `Annotated` beside the allowed values, such as
    `stock: Annotated[float, RealSet.closed_open(0, oo), Quantity('money')] = 100.0`. The total of a quantity changes
    only by its flows with the environment, so the conservation monitor checks it at every frame, and a law other than a
    flow cannot change a value that holds it.

    Unlike the allowed values in `Annotated`, which say what a value may be, it says what a value holds. Choose it when
    a total must change only by flows, such as money that moves between accounts.

    It appears in many fields. The money of a bank, the people of an epidemic, and the energy of a reserve are each a
    quantity.

    Args:
        name:
            The name of the quantity, such as `'money'`.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element, Quantity
        >>> class Account(Element):
        ...     def __init__(self, label, stock: Annotated[float, Quantity('money')] = 100.0):
        ...         super().__init__(label)
        ...         self.stock = stock
        >>> Quantity('money').name
        'money'
    """

    name: str

    def _check(self) -> None:
        """Raise an error for a name that is not a nonempty string.

        Raises:
            ValidationError:
                The name is not a nonempty string.
        """
        if not isinstance(self.name, str) or not self.name:
            error_msg = f"A quantity has a nonempty name, such as Quantity('energy'), but it is {self.name!r}."
            raise ValidationError(error_msg)
