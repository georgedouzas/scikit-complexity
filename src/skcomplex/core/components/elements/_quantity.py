"""Define the quantity a value of an element holds."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...base import BaseSpec
from ...errors import ValidationError


@dataclass(frozen=True)
class Quantity(BaseSpec):
    """A spec of a value that holds a conserved quantity, such as money.

    It is written in `Annotated` beside the allowed values, such as `stock: Annotated[float, Quantity('money')] = 100.0`.
    The total of a quantity changes only by its flows to and from the outside, so the conservation observer checks it at
    every frame, and a law other than a flow cannot change a value that holds it. It holds the money of a bank, the people
    of an epidemic, and the energy of a reserve.

    Read more in the user guide, under Elements.

    Args:
        name:
            The name of the quantity, such as `'money'`.

    See Also:
        Member: Marks a value that names a member of another group.
        ConservationObserver: Checks that the totals of quantities hold.

    Examples:
        >>> from skcomplex.core.components.elements import Quantity
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
