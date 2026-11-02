"""Define compartments, elements that hold the amount of units in one named state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ...specs import StateSpec, VisualSpec
from ._element import Element


class CompartmentElement(Element):
    """Compartment, the amount of units in one named state, as a count or a fraction.

    The label of the element names the compartment, such as `'A'`, `'B'`, or `'source'`. A compartment has no laws
    of its own, and reactions and flows between compartments move its units.

    Args:
        label:
            The label of the compartment, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n:
            The initial amount, a nonnegative count or fraction.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

    Examples:
        >>> from skcomplex.core.elements import CompartmentElement
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> system = System(elements=[CompartmentElement('A', n=990), CompartmentElement('B', n=10)])
        >>> Simulation(system, dt=1.0).check().prepare().run(until=2).trajectory_.values('A.n').tolist()
        [990.0, 990.0, 990.0]
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n
