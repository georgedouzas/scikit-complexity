"""Branch of a path that ants can take, holding the number `n` of pheromone marks left by the ants that took it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import CompartmentElement
from ....core.specs import StateSpec, VisualSpec


class BranchElement(CompartmentElement):
    """Branch of a path that ants can take, holding the number `n` of pheromone marks left by the ants that took it.

    Args:
        label:
            The label of the branch.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n:
            The initial number of marks.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import BranchElement
        >>> BranchElement('left', n=5.0).n
        5.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual, n=n)
