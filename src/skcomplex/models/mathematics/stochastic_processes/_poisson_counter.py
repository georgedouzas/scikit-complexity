"""Counter of the events of a Poisson process, with the rate of events it is counted at."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class PoissonCounterElement(Element):
    """Counter of the events of a Poisson process, with the rate of events it is counted at.

    Args:
        label:
            The label of the counter or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        count:
            The initial number of events.

        rate:
            The rate of events, which the jump of the counter reads.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import PoissonCounterElement
        >>> PoissonCounterElement('counter', rate=2.0).rate
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        count: Annotated[float, StateSpec(domain=NN)] = 0.0,
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.count = count
        self.rate = rate
