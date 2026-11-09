"""Define compartments, elements that hold the amount of units in one named state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.elements import NonnegativeElement
from .....core.specs import StateSpec, VisualSpec


class CompartmentElement(NonnegativeElement):
    """Compartment, the amount `x` of units in one named state, as a count or a fraction.

    The label of the element names the compartment, such as `'S'`, `'I'`, or `'plasma'`. A compartment has no laws of
    its own, and the reactions and flows of epidemics, pharmacokinetics, and populations move its units.

    Args:
        label:
            The label of the compartment, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial amount, a nonnegative count or fraction.

    Examples:
        >>> from skcomplex.core.components.systems import System
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.biology.common.kinetics import CompartmentElement
        >>> system = System(elements=[CompartmentElement('S', x=990), CompartmentElement('I', x=10)])
        >>> simulate(system, until=2, dt=1.0).values('S.x').tolist()
        [990.0, 990.0, 990.0]
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual, x=x)
