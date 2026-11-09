"""Define nonnegative elements, elements that hold one nonnegative scalar."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class NonnegativeElement(Element):
    """Element that holds one nonnegative scalar `x`.

    The scalar is an amount, a fraction, a density at one class, or a concentration. The element has no laws of its
    own, and the reactions, flows, and encounters of its system change it. A group of these elements on the sites of a
    lattice of classes is a distribution over the classes.

    Args:
        label:
            The label of the element, unique among the parts of its system.

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
        >>> simulate(system, until=2, dt=1.0).values('A.x').tolist()
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
