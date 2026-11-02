"""Walker on the integers with a direction `d = ±1`, which it keeps with the probability `persistence`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import ZZ, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class PersistentWalkerElement(Element):
    """Walker on the integers with a direction `d = ±1`, which it keeps with the probability `persistence`.

    Args:
        label:
            The label of the walker or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        d:
            The initial direction, 1 or -1.

        x:
            The initial position.

        persistence:
            The probability of keeping the direction at a step.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import PersistentWalkerElement
        >>> PersistentWalkerElement('walker', persistence=0.9).persistence
        0.9
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        d: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
        x: Annotated[float, StateSpec(domain=ZZ, role='position')] = 0.0,
        persistence: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
    ) -> None:
        super().__init__(label, visual=visual)
        self.d = d
        self.x = x
        self.persistence = persistence
