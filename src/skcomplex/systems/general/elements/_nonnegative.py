"""Define the element that holds one nonnegative number."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.spaces import BaseSpace


class NonnegativeElement(Element):
    """An element that holds one nonnegative number `x`.

    The number is an amount, a share, a concentration, or the density at one class of a distribution. The reactions,
    flows, and encounters of its system change it, and a value below zero is rejected. With `size` it holds one number
    per member, and with `space` one number per site, such as a distribution over the classes of a lattice.

    Unlike `Element`, which takes any values, it fixes one value `x` at zero or above. Choose it when a system tracks an
    amount that cannot be negative, such as a count, a mass, or a sum of money.

    It appears in many fields. The number of people infected in an epidemic, the moles of a chemical species in a
    reactor, the money in a bank account, and the tokens in a place of a Petri net are each one nonnegative number.

    Args:
        label:
            The label of the element, unique among the components of its system.

        x:
            The initial value, a nonnegative number, or one per member or site.

        size:
            The number of members, or `None` for one.

        space:
            The space with one member per site, or `None` for none.

    Attributes:
        default_visual_:
            How the element is drawn by default, the `visual` of its class.

    Examples:
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.general.elements import NonnegativeElement
        >>> System(elements=[NonnegativeElement('A', x=990)]).prepare().simulate(until=2).get_values('A.x').tolist()
        [990.0]
        >>> System(elements=[NonnegativeElement('A', x=-1.0)]).prepare()
        Traceback (most recent call last):
        ...
        skcomplex.core.errors.ValidationError: ...
    """

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.x = x
