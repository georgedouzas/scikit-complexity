"""Define the element that holds one nonnegative number."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class NonnegativeElement(Element):
    """An element that holds one nonnegative number `x`.

    The number is an amount, a share, a concentration, or the density at one class of a distribution. The element has no
    laws of its own, and the reactions, flows, and encounters of its system change it. A group of these elements on the
    sites of a lattice of classes is a distribution over the classes.

    Unlike `DeclaredElement`, which takes its states as data, it fixes one state `x` at zero or above. Choose it when a
    system tracks an amount that cannot be negative, such as a count, a mass, or a sum of money.

    It appears in many fields. The number of people infected in an epidemic, the moles of a chemical species in a
    reactor, the money in a bank account, and the tokens in a place of a Petri net are each one nonnegative number.

    Args:
        label:
            The label of the element, unique among the components of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value, a nonnegative number.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

    Examples:
        >>> from skcomplex.systems.general.elements import NonnegativeElement
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> system = CompositeSystem(elements=[NonnegativeElement('A', x=990), NonnegativeElement('B', x=10)])
        >>> simulate(system, until=2, dt=1.0).get_values('A.x').tolist()
        [990.0, 990.0, 990.0]
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
